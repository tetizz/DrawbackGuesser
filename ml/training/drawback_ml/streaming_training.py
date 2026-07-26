"""Production bounded-memory training over a repeatedly readable example stream."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
import ctypes
import errno
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
from typing import Any

from .checkpoint import (
    file_sha256,
    save_checkpoint,
    write_checkpoint_index,
    write_run_metadata,
)
from .features import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    MOVE_VOCABULARY_SIZE,
    build_feature_vector,
    encode_move,
)
from .model import (
    HYBRID_MODEL_VARIANTS,
    SEQUENCE_MODEL_VARIANTS,
    ModelConfig,
    create_model,
    create_sequence_model,
)
from .parameters import (
    ParameterVocabulary,
    encode_parameter_targets,
    supervised_parameter_label,
)
from .records import TrainingExample
from .sequence import (
    MASKED_CURRENT_MOVE_TOKEN,
    PAD_TOKEN,
    UNKNOWN_CURRENT_MOVE_TOKEN,
    UNKNOWN_TOKEN,
    ObservationTokenizerV2,
    SanTokenizer,
    SequenceTokenizer,
    current_move_token,
    encode_public_sequence,
)
from .streaming import (
    build_player_game_sampling_plan,
    deterministic_shuffle_buffer,
    iter_batches,
    player_game_balanced_examples,
)
from .symbolic import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
    build_symbolic_feature_vector,
    fusion_aware_drawback_loss,
    fusion_aware_loss_metadata,
)
from .training import (
    TrainingConfig,
    balanced_legal_mask_loss,
    drawback_observation_masks,
)


ExampleFactory = Callable[[], Iterator[TrainingExample]]
FinalValidation = Callable[[Path], None]
MAX_STREAMING_VOCABULARY = 65_536
AT_FDCWD = -100
RENAME_NOREPLACE = 1
RENAME_EXCL = 0x00000004


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory, failing if the destination exists."""

    if os.name == "nt":
        source.rename(destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace directory publication is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            AT_FDCWD,
            source_bytes,
            AT_FDCWD,
            destination_bytes,
            RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise RuntimeError("atomic no-replace directory publication is unavailable")
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, RENAME_EXCL)
    else:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported "
            f"on {sys.platform}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                str(destination),
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


@contextmanager
def staged_output_directory(output_directory: Path) -> Iterator[Path]:
    """Yield a private same-parent directory and publish it without clobbering."""

    if output_directory.exists():
        raise FileExistsError(
            f"training output directory already exists: {output_directory}"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    try:
        yield staging_directory
        try:
            _rename_directory_no_replace(staging_directory, output_directory)
        except FileExistsError as error:
            raise FileExistsError(
                f"training output directory already exists: {output_directory}"
            ) from error
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)


def validate_training_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested != "cuda":
        raise ValueError("training device must be cpu or cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    major, minor = torch.cuda.get_device_capability(0)
    architecture = f"sm_{major}{minor}"
    compiled = set(torch.cuda.get_arch_list())
    if architecture not in compiled:
        raise RuntimeError(
            f"CUDA device requires {architecture}, but this PyTorch build "
            f"contains only {', '.join(sorted(compiled)) or 'no CUDA architectures'}"
        )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic CUDA training requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    return "cuda"


def _runtime_metadata(torch: Any, device: str) -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": device,
        "deterministic_algorithms": True,
    }
    if device == "cuda":
        result.update(
            {
                "cuda": str(torch.version.cuda),
                "cudnn": int(torch.backends.cudnn.version() or 0),
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            }
        )
    return result


def _claim_run(
    directory: Path,
    config: TrainingConfig,
    runtime: dict[str, object],
    sampling: dict[str, object],
) -> str:
    config_material = asdict(config)
    if config.sequence_observation_mode is None:
        config_material.pop("sequence_observation_mode")
    material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": config_material,
        "runtime": runtime,
        "sampling": sampling,
    }
    run_id = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / "run.claim.json"
    payload = json.dumps({"run_id": run_id, **material}, indent=2, sort_keys=True) + "\n"
    try:
        with claim.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as target:
            target.write(payload)
    except FileExistsError as error:
        raise FileExistsError(
            f"training output directory is already claimed: {directory}"
        ) from error
    return run_id


def _scan(
    factory: ExampleFactory,
    required_vocabulary: tuple[str, ...],
    *,
    max_history: int,
    model_variant: str | None = None,
    sequence: bool | None = None,
) -> tuple[ParameterVocabulary, SequenceTokenizer | None, int, int]:
    if model_variant is None:
        if not isinstance(sequence, bool):
            raise ValueError("scan requires a model variant or sequence mode")
        model_variant = "v2-gru" if sequence else "v1"
    elif sequence is not None:
        raise ValueError("scan cannot combine model_variant and sequence")
    required = set(required_vocabulary)
    white: set[str] = set()
    black: set[str] = set()
    parameters: set[str] = set()
    san_tokens: set[str] = set()
    count = 0
    game_count = 0
    previous_game: str | None = None
    for example in factory():
        count += 1
        if example.game_id != previous_game:
            game_count += 1
            previous_game = example.game_id
        white.add(example.white_drawback)
        black.add(example.black_drawback)
        for value in (example.white_parameters, example.black_parameters):
            supervised = supervised_parameter_label(value)
            if supervised is not None:
                parameters.add(supervised)
        if model_variant in SEQUENCE_MODEL_VARIANTS:
            san_tokens.update(example.features.history_san)
        if model_variant == "v22-hybrid":
            san_tokens.add(current_move_token(example.features.move))
        if (
            len(parameters) > MAX_STREAMING_VOCABULARY
            or len(san_tokens) > MAX_STREAMING_VOCABULARY
        ):
            raise ValueError("streaming vocabulary exceeds its bounded limit")
    if count == 0:
        raise ValueError("authenticated training split has no move examples")
    unexpected = (white | black) - required
    missing_white = required - white
    missing_black = required - black
    if unexpected or missing_white or missing_black:
        raise ValueError(
            "required drawback coverage is incomplete or contains unknown labels"
        )
    parameter_vocabulary = ParameterVocabulary.build(parameters)
    if model_variant == "v22-hybrid":
        tokenizer: SequenceTokenizer | None = ObservationTokenizerV2(
            (
                PAD_TOKEN,
                UNKNOWN_TOKEN,
                UNKNOWN_CURRENT_MOVE_TOKEN,
                MASKED_CURRENT_MOVE_TOKEN,
                *sorted(san_tokens),
            ),
            max_history + 1,
        )
    elif model_variant in SEQUENCE_MODEL_VARIANTS:
        tokenizer = SanTokenizer(
            ("<pad>", "<unk>", *sorted(san_tokens)),
            max_history,
        )
    else:
        tokenizer = None
    return parameter_vocabulary, tokenizer, count, game_count


def train_streaming_baseline(
    factory: ExampleFactory,
    output_directory: Path,
    config: TrainingConfig,
    *,
    final_validation: FinalValidation | None = None,
) -> Path:
    """Train while deterministically releasing temporary sampling storage."""

    with ExitStack() as resources:
        return _train_streaming_baseline(
            factory,
            output_directory,
            config,
            final_validation=final_validation,
            resources=resources,
        )


def _train_streaming_baseline(
    factory: ExampleFactory,
    output_directory: Path,
    config: TrainingConfig,
    *,
    final_validation: FinalValidation | None,
    resources: ExitStack,
) -> Path:
    """Train without retaining corpus-sized Python collections."""

    if config.required_drawback_vocabulary is None:
        raise ValueError("streaming training requires a frozen drawback vocabulary")
    if output_directory.exists():
        raise FileExistsError(
            f"training output directory already exists: {output_directory}"
        )
    vocabulary = list(config.required_drawback_vocabulary)
    class_index = {label: index for index, label in enumerate(vocabulary)}
    sequence = config.model_variant in SEQUENCE_MODEL_VARIANTS
    parameter_vocabulary, tokenizer, raw_example_count, game_count = _scan(
        factory,
        tuple(vocabulary),
        model_variant=config.model_variant,
        max_history=config.max_history,
    )
    player_game_plan = resources.enter_context(
        build_player_game_sampling_plan(
            factory(),
            labels=tuple(vocabulary),
        )
    )
    if (
        player_game_plan.raw_examples != raw_example_count
        or player_game_plan.row_bearing_games != game_count
    ):
        raise RuntimeError("training corpus changed between streaming passes")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for streaming training") from error
    device = validate_training_device(torch, config.device)
    runtime = _runtime_metadata(torch, device)
    if config.player_game_examples_per_epoch is None:
        if config.game_examples_per_epoch % 2:
            raise ValueError(
                "legacy game_examples_per_epoch must be even for equal "
                "observed-player balancing; use player_game_examples_per_epoch"
            )
        examples_per_player_game = config.game_examples_per_epoch // 2
        examples_setting = "legacy-game-examples-divided-by-two"
    else:
        examples_per_player_game = config.player_game_examples_per_epoch
        examples_setting = "explicit-player-game-examples"
    sampling = player_game_plan.metadata(examples_per_player_game)
    sampling.update(
        {
            "examples_setting": examples_setting,
            "configured_game_examples_per_epoch": (
                config.game_examples_per_epoch
            ),
            "configured_player_game_examples_per_epoch": (
                config.player_game_examples_per_epoch
            ),
        }
    )
    effective_examples = int(sampling["effective_examples_per_epoch"])
    with staged_output_directory(output_directory) as staging_directory:
        run_id = _claim_run(staging_directory, config, runtime, sampling)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        if device == "cuda":
            torch.backends.cudnn.benchmark = False

        model_config = ModelConfig(
            input_dimension=FEATURE_DIMENSION,
            drawback_classes=len(vocabulary),
            legal_mask_dimension=MOVE_VOCABULARY_SIZE,
            parameter_classes=len(parameter_vocabulary.tokens),
            hidden_dimension=config.hidden_dimension,
            model_variant=config.model_variant,
            sequence_observation_mode=config.sequence_observation_mode,
            san_vocabulary_size=(
                None if tokenizer is None else len(tokenizer.vocabulary)
            ),
            san_embedding_dimension=config.san_embedding_dimension,
            sequence_hidden_dimension=config.sequence_hidden_dimension,
            symbolic_dimension=(
                SYMBOLIC_FEATURE_DIMENSION
                if config.model_variant in HYBRID_MODEL_VARIANTS
                else None
            ),
            symbolic_hidden_dimension=config.symbolic_hidden_dimension,
        )
        model = (
            create_sequence_model(model_config) if sequence else create_model(model_config)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        classification_loss = torch.nn.CrossEntropyLoss()
        binary_loss = torch.nn.BCEWithLogitsLoss()
        checkpoint_sha256s: list[str] = []

        for epoch in range(1, config.epochs + 1):
            balanced = player_game_balanced_examples(
                factory(),
                plan=player_game_plan,
                seed=config.seed,
                epoch=epoch,
                examples_per_player_game=examples_per_player_game,
            )
            ordered = deterministic_shuffle_buffer(
                balanced,
                seed=config.seed + epoch,
                buffer_size=config.shuffle_buffer_size,
            )
            seen = 0
            model.train()
            for batch in iter_batches(ordered, config.batch_size):
                seen += len(batch)
                inputs = torch.tensor(
                    [build_feature_vector(item.features) for item in batch],
                    dtype=torch.float32,
                    device=device,
                )
                white = torch.tensor(
                    [class_index[item.white_drawback] for item in batch],
                    dtype=torch.long,
                    device=device,
                )
                black = torch.tensor(
                    [class_index[item.black_drawback] for item in batch],
                    dtype=torch.long,
                    device=device,
                )
                white_observed, black_observed = drawback_observation_masks(
                    tuple(item.features.player_color for item in batch),
                    tuple(len(item.features.history_san) for item in batch),
                )
                white_drawback_mask = torch.tensor(
                    white_observed, dtype=torch.bool, device=device
                )
                black_drawback_mask = torch.tensor(
                    black_observed, dtype=torch.bool, device=device
                )
                targets = encode_parameter_targets(
                    parameter_vocabulary,
                    (item.white_parameters for item in batch),
                    (item.black_parameters for item in batch),
                )
                white_parameters = torch.tensor(
                    targets.white_indices, dtype=torch.long, device=device
                )
                black_parameters = torch.tensor(
                    targets.black_indices, dtype=torch.long, device=device
                )
                white_parameter_mask = torch.tensor(
                    targets.white_mask, dtype=torch.bool, device=device
                )
                black_parameter_mask = torch.tensor(
                    targets.black_mask, dtype=torch.bool, device=device
                )
                white_parameter_mask &= white_drawback_mask
                black_parameter_mask &= black_drawback_mask
                trigger = torch.tensor(
                    [float(item.rule_triggered) for item in batch],
                    dtype=torch.float32,
                    device=device,
                )
                legal_mask = torch.zeros(
                    (len(batch), MOVE_VOCABULARY_SIZE),
                    dtype=torch.float32,
                    device=device,
                )
                for row_index, item in enumerate(batch):
                    for move in item.drawback_legal_moves:
                        legal_mask[row_index, encode_move(move)] = 1.0
                if tokenizer is None:
                    outputs = model(inputs)
                else:
                    histories = [
                        encode_public_sequence(
                            tokenizer,
                            item.features,
                            config.sequence_observation_mode,
                        )
                        for item in batch
                    ]
                    tokens = torch.tensor(
                        [value for value, _ in histories],
                        dtype=torch.long,
                        device=device,
                    )
                    lengths = torch.tensor(
                        [length for _, length in histories],
                        dtype=torch.long,
                        device=device,
                    )
                    if config.model_variant in HYBRID_MODEL_VARIANTS:
                        symbolic = torch.tensor(
                            [build_symbolic_feature_vector(item.features) for item in batch],
                            dtype=torch.float32,
                            device=device,
                        )
                        outputs = model(inputs, tokens, lengths, symbolic)
                    else:
                        outputs = model(inputs, tokens, lengths)
                parameter_loss = outputs["white_parameters"].sum() * 0.0
                if white_parameter_mask.any():
                    parameter_loss += classification_loss(
                        outputs["white_parameters"][white_parameter_mask],
                        white_parameters[white_parameter_mask],
                    )
                if black_parameter_mask.any():
                    parameter_loss += classification_loss(
                        outputs["black_parameters"][black_parameter_mask],
                        black_parameters[black_parameter_mask],
                    )
                legal_loss = (
                    balanced_legal_mask_loss(torch, outputs["legal_mask"], legal_mask)
                    if sequence
                    else binary_loss(outputs["legal_mask"], legal_mask)
                )
                drawback_loss = outputs["white_drawback"].sum() * 0.0
                if config.model_variant in HYBRID_MODEL_VARIANTS:
                    drawback_loss += fusion_aware_drawback_loss(
                        torch,
                        outputs["white_drawback"],
                        [item.features for item in batch],
                        vocabulary,
                        "white",
                        white,
                        white_drawback_mask,
                    )
                    drawback_loss += fusion_aware_drawback_loss(
                        torch,
                        outputs["black_drawback"],
                        [item.features for item in batch],
                        vocabulary,
                        "black",
                        black,
                        black_drawback_mask,
                    )
                else:
                    if white_drawback_mask.any():
                        drawback_loss += classification_loss(
                            outputs["white_drawback"][white_drawback_mask],
                            white[white_drawback_mask],
                        )
                    if black_drawback_mask.any():
                        drawback_loss += classification_loss(
                            outputs["black_drawback"][black_drawback_mask],
                            black[black_drawback_mask],
                        )
                loss = (
                    drawback_loss
                    + config.trigger_loss_weight
                    * binary_loss(outputs["trigger"], trigger)
                    + config.legal_mask_loss_weight * legal_loss
                    + config.parameter_loss_weight * parameter_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if seen != effective_examples:
                raise RuntimeError("training corpus changed between streaming passes")
            saved_checkpoint = save_checkpoint(
                staging_directory,
                model=model,
                optimizer=optimizer,
                seed=config.seed,
                epoch=epoch,
                drawback_vocabulary=vocabulary,
                parameter_vocabulary=list(parameter_vocabulary.tokens),
                model_config={
                    "input_dimension": model_config.input_dimension,
                    "drawback_classes": model_config.drawback_classes,
                    "parameter_classes": model_config.parameter_classes,
                    "legal_mask_dimension": model_config.legal_mask_dimension,
                    "hidden_dimension": model_config.hidden_dimension,
                    "model_variant": model_config.model_variant,
                    **(
                        {
                            "sequence_observation_mode": (
                                model_config.sequence_observation_mode
                            )
                        }
                        if model_config.sequence_observation_mode is not None
                        else {}
                    ),
                    "san_vocabulary_size": model_config.san_vocabulary_size,
                    "san_embedding_dimension": model_config.san_embedding_dimension,
                    "sequence_hidden_dimension": model_config.sequence_hidden_dimension,
                    "symbolic_dimension": model_config.symbolic_dimension,
                    "symbolic_hidden_dimension": model_config.symbolic_hidden_dimension,
                },
                training_metadata={
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "model_variant": config.model_variant,
                    **(
                        {
                            "sequence_observation_mode": (
                                config.sequence_observation_mode
                            )
                        }
                        if config.sequence_observation_mode is not None
                        else {}
                    ),
                    "batch_size": config.batch_size,
                    "shuffle_buffer_size": config.shuffle_buffer_size,
                    "streaming_order": "seeded-buffer-v1",
                    "sampling": sampling,
                    "loss_weights": {
                        "drawback": 1.0,
                        "trigger": config.trigger_loss_weight,
                        "legal_mask": config.legal_mask_loss_weight,
                        "parameter": config.parameter_loss_weight,
                    },
                    "symbolic_feature_version": (
                        SYMBOLIC_FEATURE_VERSION
                        if config.model_variant in HYBRID_MODEL_VARIANTS
                        else None
                    ),
                    "symbolic_rule_ids": (
                        list(SYMBOLIC_RULE_IDS)
                        if config.model_variant in HYBRID_MODEL_VARIANTS
                        else None
                    ),
                    "drawback_loss_objective": fusion_aware_loss_metadata(
                        config.model_variant
                    ),
                    "san_tokenizer": (
                        None if tokenizer is None else tokenizer.metadata()
                    ),
                    "corpus_provenance": (
                        None
                        if config.corpus_provenance is None
                        else dict(config.corpus_provenance)
                    ),
                    "runtime": runtime,
                    "run_id": run_id,
                },
            )
            checkpoint_sha256s.append(file_sha256(saved_checkpoint))
        write_run_metadata(
            staging_directory,
            {
                "format_version": 4,
                "run_id": run_id,
                "seed": config.seed,
                "epochs": config.epochs,
                "raw_rows": raw_example_count,
                "row_bearing_games": game_count,
                "effective_examples_per_epoch": effective_examples,
                "drawback_vocabulary": vocabulary,
                "parameter_vocabulary": list(parameter_vocabulary.tokens),
                "model_variant": config.model_variant,
                **(
                    {
                        "sequence_observation_mode": (
                            config.sequence_observation_mode
                        )
                    }
                    if config.sequence_observation_mode is not None
                    else {}
                ),
                "drawback_loss_objective": fusion_aware_loss_metadata(
                    config.model_variant
                ),
                "san_tokenizer": (
                    None if tokenizer is None else tokenizer.metadata()
                ),
                "corpus_provenance": (
                    None
                    if config.corpus_provenance is None
                    else dict(config.corpus_provenance)
                ),
                "runtime": runtime,
                "streaming_order": "seeded-buffer-v1",
                "shuffle_buffer_size": config.shuffle_buffer_size,
                "sampling": sampling,
                "loss_weights": {
                    "drawback": 1.0,
                    "trigger": config.trigger_loss_weight,
                    "legal_mask": config.legal_mask_loss_weight,
                    "parameter": config.parameter_loss_weight,
                },
            },
        )
        write_checkpoint_index(
            staging_directory,
            seed=config.seed,
            epochs=config.epochs,
            checkpoint_sha256s=tuple(checkpoint_sha256s),
        )
        if final_validation is not None:
            final_validation(staging_directory)
    return output_directory
