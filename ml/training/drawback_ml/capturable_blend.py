"""Authenticated validation CLI for the preregistered capturable blend."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Protocol, Sequence

from .capturable_baseline import (
    _canonical_json,
    _publish_bytes,
    tensorize,
)
from .capturable_blend_contract import (
    BLEND_WEIGHTS,
    CAPTURABLE_BLEND_FORMAT,
    CAPTURABLE_BLEND_VERSION,
    PROTOCOL_COMMIT,
    PROTOCOL_FILE,
    PROTOCOL_SHA256,
    SELECTION_METRIC,
    ComponentPredictions,
    _EXPECTED_INPUTS,
    _is_sha256,
    blend_components,
    blend_reliability_checks,
    candidate_order,
    component_predictions,
    evaluate_predictions,
    performance_order,
)
from .capturable_candidate_selection import (
    _selection_report,
    _validated_candidate,
    load_treatment_comparison,
)
from .capturable_experiment import (
    _load_selection_checkpoint,
    _load_stable_capturable_dataset,
)
from .capturable_records import CapturableDatasetError


_CANONICAL_REPOSITORY_URL = "https://github.com/tetizz/DrawbackGuesser.git"
_CANONICAL_COMMIT_IDENTITY = (
    "tetizz",
    "104690265+tetizz@users.noreply.github.com",
    "tetizz",
    "104690265+tetizz@users.noreply.github.com",
)
_WINDOWS_GIT_SHA256 = (
    "e1e5f04e40a003b28b1e79659fabf3fd04dc4d8fdc7221d3495dfe51f861c75e"
)
_LOCAL_CONFIG_REDIRECTION_PATTERN = (
    r"^(include\.path|includeif\..*\.path|"
    r"url\..*\.(insteadof|pushinsteadof)|"
    r"http\..*|credential\..*|filter\..*|"
    r"core\.(attributesfile|excludesfile|fsmonitor|sshcommand)|"
    r"remote\..*\.(proxy|proxycommand|receivepack|uploadpack))$"
)


class _GitRunner(Protocol):
    def __call__(
        self,
        *arguments: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        ...


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return path.is_symlink()
    return path.is_symlink() or (
        os.name == "nt"
        and bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
        )
    )


def _path_identity(
    path: Path,
    *,
    directory: str | None,
    filename: str,
    label: str,
) -> Path:
    resolved = path.resolve()
    if resolved.name != filename or (
        directory is not None and resolved.parent.name != directory
    ):
        raise CapturableDatasetError(f"{label} path is not preregistered")
    return resolved


def _is_git_revision(value: str) -> bool:
    return len(value) == 40 and all(
        token in "0123456789abcdef" for token in value
    )


def _windows_known_directory(kind: str) -> Path:
    if os.name != "nt":
        raise CapturableDatasetError(
            "Windows known-directory lookup is unavailable"
        )
    buffer = ctypes.create_unicode_buffer(32_768)
    if kind == "program-files":
        result = ctypes.windll.shell32.SHGetFolderPathW(
            None,
            0x0026,
            None,
            0,
            buffer,
        )
        if result != 0:
            raise CapturableDatasetError(
                "trusted Windows Program Files directory is unavailable"
            )
    else:
        function_name = {
            "system": "GetSystemDirectoryW",
            "windows": "GetSystemWindowsDirectoryW",
        }.get(kind)
        if function_name is None:
            raise CapturableDatasetError(
                "Windows known-directory role is invalid"
            )
        length = getattr(
            ctypes.windll.kernel32,
            function_name,
        )(buffer, len(buffer))
        if length <= 0 or length >= len(buffer):
            raise CapturableDatasetError(
                f"trusted Windows {kind} directory is unavailable"
            )
    try:
        path = Path(buffer.value).resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            f"trusted Windows {kind} directory is unavailable"
        ) from error
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise CapturableDatasetError(
            f"trusted Windows {kind} directory is invalid"
        )
    return path


def _trusted_git_executable() -> Path:
    if os.name == "nt":
        candidate = (
            _windows_known_directory("program-files")
            / "Git"
            / "cmd"
            / "git.exe"
        )
    else:
        candidates = {
            path.resolve(strict=True)
            for path in (Path("/usr/bin/git"), Path("/bin/git"))
            if path.is_file()
        }
        if len(candidates) != 1:
            raise CapturableDatasetError(
                "trusted system Git executable is unavailable"
            )
        candidate = candidates.pop()
    try:
        executable = candidate.resolve(strict=True)
        payload = executable.read_bytes()
    except OSError as error:
        raise CapturableDatasetError(
            "trusted system Git executable is unavailable"
        ) from error
    if not executable.is_file() or _is_link_or_reparse_point(executable):
        raise CapturableDatasetError(
            "trusted system Git executable is invalid"
        )
    if (
        os.name == "nt"
        and hashlib.sha256(payload).hexdigest() != _WINDOWS_GIT_SHA256
    ):
        raise CapturableDatasetError(
            "trusted Windows Git executable identity changed"
        )
    return executable


def _authenticated_git_environment(git_executable: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"TEMP", "TMP"}
    }
    if os.name == "nt":
        windows_directory = _windows_known_directory("windows")
        system_directory = _windows_known_directory("system")
        shell = system_directory / "cmd.exe"
        if not shell.is_file() or _is_link_or_reparse_point(shell):
            raise CapturableDatasetError(
                "trusted Windows command shell is unavailable"
            )
        environment.update(
            {
                "ComSpec": str(shell.resolve(strict=True)),
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                "PATH": os.pathsep.join(
                    (
                        str(git_executable.parent),
                        str(system_directory),
                        str(windows_directory),
                    )
                ),
                "SystemRoot": str(windows_directory),
                "WINDIR": str(windows_directory),
            }
        )
    else:
        environment["PATH"] = os.pathsep.join(("/usr/bin", "/bin"))
    environment.update(
        {
            "GCM_INTERACTIVE": "Never",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _authenticated_git_runner(
    repository: Path,
) -> _GitRunner:
    git_executable = _trusted_git_executable()
    git_environment = _authenticated_git_environment(git_executable)

    def git(
        *arguments: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(git_executable),
                "--no-replace-objects",
                "-c",
                "core.attributesFile=",
                "-c",
                "core.autocrlf=true",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.askPass=",
                "-c",
                "credential.helper=",
                "-c",
                "credential.interactive=false",
                "-c",
                "http.sslVerify=true",
                "-c",
                "http.proxy=",
                "-c",
                "http.https://github.com/.proxy=",
                "-c",
                "http.https://github.com/.sslVerify=true",
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.https.allow=always",
                *arguments,
            ],
            cwd=repository,
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=git_environment,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )

    return git


def _validated_protocol_identity_inputs(
    *,
    protocol_commit: str,
    protocol_file: str,
    protocol_sha256: str,
    operation: str,
) -> None:
    if (
        not _is_git_revision(protocol_commit)
        or not _is_sha256(protocol_sha256)
        or not protocol_file
        or Path(protocol_file).name != protocol_file
        or "/" in protocol_file
        or "\\" in protocol_file
    ):
        raise CapturableDatasetError(
            f"{operation} protocol identity is not canonical"
        )


def _repository_anchor(
    *,
    git: _GitRunner,
    repository: Path,
) -> tuple[bool, list[str], subprocess.CompletedProcess[str]]:
    top_level_matches = (
        Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
        == repository
    )
    origin_urls = git(
        "config",
        "--local",
        "--no-includes",
        "--get-all",
        "remote.origin.url",
    ).stdout.splitlines()
    configured_redirections = git(
        "config",
        "--no-includes",
        "--show-scope",
        "--name-only",
        "--get-regexp",
        _LOCAL_CONFIG_REDIRECTION_PATTERN,
        check=False,
    )
    if configured_redirections.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            configured_redirections.returncode,
            configured_redirections.args,
            output=configured_redirections.stdout,
            stderr=configured_redirections.stderr,
        )
    unsafe_redirections = [
        line
        for line in configured_redirections.stdout.splitlines()
        if line.startswith("local\t") or line.startswith("worktree\t")
    ]
    local_redirections = subprocess.CompletedProcess(
        configured_redirections.args,
        0 if unsafe_redirections else 1,
        "\n".join(unsafe_redirections),
        configured_redirections.stderr,
    )
    return top_level_matches, origin_urls, local_redirections


def _remote_main_fields(
    git: _GitRunner,
) -> list[str]:
    return git(
        "ls-remote",
        _CANONICAL_REPOSITORY_URL,
        "refs/heads/main",
    ).stdout.split()


def _authenticated_execution_identity(
    *,
    protocol_commit: str,
    protocol_file: str,
    protocol_sha256: str,
    operation: str,
) -> Mapping[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    protocol_path = repository / "docs" / "research" / protocol_file
    _validated_protocol_identity_inputs(
        protocol_commit=protocol_commit,
        protocol_file=protocol_file,
        protocol_sha256=protocol_sha256,
        operation=operation,
    )
    git = _authenticated_git_runner(repository)

    try:
        measured_protocol_sha256 = hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest()
        # `git status` can launch repository-configured clean/process filters.
        # Reject every command-bearing local/worktree redirection before status.
        top_level_matches, origin_urls, local_redirections = _repository_anchor(
            git=git,
            repository=repository,
        )
        if local_redirections.stdout:
            raise CapturableDatasetError(
                f"{operation} repository identity is not the pushed release"
            )
        status = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        revision = git("rev-parse", "HEAD").stdout.strip()
        if not _is_git_revision(revision):
            raise CapturableDatasetError(
                f"{operation} revision does not contain the protocol"
            )
        ancestor = git(
            "merge-base",
            "--is-ancestor",
            protocol_commit,
            revision,
            check=False,
        )
        committed_protocol = git(
            "show",
            f"{protocol_commit}:docs/research/{protocol_file}",
        ).stdout.encode()
        remote_revision_fields = _remote_main_fields(git)
        commit_identity = git(
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
            revision,
        ).stdout.rstrip("\n").split("\x00")
    except CapturableDatasetError:
        raise
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
        ValueError,
    ) as error:
        raise CapturableDatasetError(
            f"{operation} execution identity cannot be verified"
        ) from error
    if measured_protocol_sha256 != protocol_sha256:
        raise CapturableDatasetError(
            f"the committed {operation} protocol bytes have changed"
        )
    if status.stdout:
        raise CapturableDatasetError(
            f"{operation} requires a clean committed worktree"
        )
    if (
        ancestor.returncode != 0
    ):
        raise CapturableDatasetError(
            f"{operation} revision does not contain the protocol"
        )
    if (
        not top_level_matches
        or hashlib.sha256(committed_protocol).hexdigest()
        != protocol_sha256
        or origin_urls != [_CANONICAL_REPOSITORY_URL]
        or remote_revision_fields
        != [revision, "refs/heads/main"]
        or tuple(commit_identity) != _CANONICAL_COMMIT_IDENTITY
    ):
        raise CapturableDatasetError(
            f"{operation} repository identity is not the pushed release"
        )
    return {
        "cleanWorktree": True,
        "repository": "DrawbackGuesser",
        "revision": revision,
    }


def _authenticated_recorded_revision_identity(
    *,
    revision: str,
    protocol_commit: str,
    protocol_file: str,
    protocol_sha256: str,
    operation: str,
) -> Mapping[str, Any]:
    """Authenticate a historical revision against the live pushed main tip."""

    repository = Path(__file__).resolve().parents[3]
    _validated_protocol_identity_inputs(
        protocol_commit=protocol_commit,
        protocol_file=protocol_file,
        protocol_sha256=protocol_sha256,
        operation=operation,
    )
    if not _is_git_revision(revision):
        raise CapturableDatasetError(
            f"{operation} recorded revision is not canonical"
        )
    git = _authenticated_git_runner(repository)

    try:
        pushed_main_revision = git("rev-parse", "HEAD").stdout.strip()
        if not _is_git_revision(pushed_main_revision):
            raise CapturableDatasetError(
                f"{operation} pushed main revision is not canonical"
            )
        top_level_matches, origin_urls, local_redirections = _repository_anchor(
            git=git,
            repository=repository,
        )
        remote_revision_fields = _remote_main_fields(git)
        reachable = git(
            "merge-base",
            "--is-ancestor",
            revision,
            pushed_main_revision,
            check=False,
        )
        contains_protocol_commit = git(
            "merge-base",
            "--is-ancestor",
            protocol_commit,
            revision,
            check=False,
        )
        committed_protocol = git(
            "show",
            f"{revision}:docs/research/{protocol_file}",
        ).stdout.encode()
        commit_identity = git(
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
            revision,
        ).stdout.rstrip("\n").split("\x00")
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
        ValueError,
    ) as error:
        raise CapturableDatasetError(
            f"{operation} recorded revision cannot be verified"
        ) from error

    if (
        not top_level_matches
        or origin_urls != [_CANONICAL_REPOSITORY_URL]
        or local_redirections.stdout
        or remote_revision_fields
        != [pushed_main_revision, "refs/heads/main"]
        or reachable.returncode != 0
        or contains_protocol_commit.returncode != 0
        or hashlib.sha256(committed_protocol).hexdigest()
        != protocol_sha256
        or tuple(commit_identity) != _CANONICAL_COMMIT_IDENTITY
    ):
        raise CapturableDatasetError(
            f"{operation} recorded revision is not on the pushed release"
        )
    return {
        "repository": "DrawbackGuesser",
        "revision": revision,
        "pushedMainRevision": pushed_main_revision,
    }


def _execution_identity() -> Mapping[str, Any]:
    return _authenticated_execution_identity(
        protocol_commit=PROTOCOL_COMMIT,
        protocol_file=PROTOCOL_FILE,
        protocol_sha256=PROTOCOL_SHA256,
        operation="blend validation",
    )


def _selection_input(
    path: Path,
    role: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    expected = _EXPECTED_INPUTS[role]
    resolved = _path_identity(
        path,
        directory=str(expected["directory"]),
        filename=str(expected["selection"]),
        label=role,
    )
    candidate, _report = _validated_candidate(resolved)
    if (
        candidate["selectionReportSha256"]
        != expected["selectionSha256"]
        or candidate["checkpointFile"] != expected["checkpoint"]
        or candidate["checkpointSha256"]
        != expected["checkpointSha256"]
    ):
        raise CapturableDatasetError(
            f"{role} selection does not match the frozen digest"
        )
    model, metadata, checkpoint_sha256 = _load_selection_checkpoint(
        resolved.parent / candidate["checkpointFile"]
    )
    if checkpoint_sha256 != expected["checkpointSha256"]:
        raise CapturableDatasetError(
            f"{role} checkpoint does not match the frozen digest"
        )
    return model, metadata, {
        "checkpoint": candidate["checkpointFile"],
        "checkpointSha256": candidate["checkpointSha256"],
        "directory": resolved.parent.name,
        "selection": resolved.name,
        "selectionSha256": candidate["selectionReportSha256"],
    }


def _load_frozen_inputs(
    control_selection_path: Path,
    treatment_selection_path: Path,
    validation_path: Path,
    prior_comparison_path: Path,
) -> tuple[
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[Any, ...],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    control_model, control_metadata, control_input = _selection_input(
        control_selection_path,
        "control",
    )
    treatment_model, treatment_metadata, treatment_input = _selection_input(
        treatment_selection_path,
        "treatment",
    )
    if (
        control_metadata["ruleIds"] != treatment_metadata["ruleIds"]
        or control_metadata["featureDimension"]
        != treatment_metadata["featureDimension"]
        or control_metadata["inputs"]["validation"]
        != treatment_metadata["inputs"]["validation"]
    ):
        raise CapturableDatasetError(
            "blend checkpoints use incompatible public contracts"
        )

    expected_validation = _EXPECTED_INPUTS["validation"]
    resolved_validation = _path_identity(
        validation_path,
        directory=None,
        filename=str(expected_validation["file"]),
        label="validation",
    )
    rows, validation_sha256 = _load_stable_capturable_dataset(
        resolved_validation
    )
    game_count = len({row.evaluation.game_id for row in rows})
    validation_identity = {
        "file": resolved_validation.name,
        "games": game_count,
        "rows": len(rows),
        "sha256": validation_sha256,
    }
    if validation_identity != expected_validation:
        raise CapturableDatasetError(
            "validation corpus does not match the frozen identity"
        )
    checkpoint_validation = {
        "path": resolved_validation.name,
        "sha256": validation_sha256,
        "rows": len(rows),
        "games": game_count,
    }
    if control_metadata["inputs"]["validation"] != checkpoint_validation:
        raise CapturableDatasetError(
            "checkpoint validation identity does not match the corpus"
        )

    expected_comparison = _EXPECTED_INPUTS["priorComparison"]
    resolved_comparison = _path_identity(
        prior_comparison_path,
        directory=None,
        filename=str(expected_comparison["file"]),
        label="prior comparison",
    )
    comparison, comparison_sha256 = load_treatment_comparison(
        resolved_comparison
    )
    if (
        comparison_sha256 != expected_comparison["sha256"]
        or comparison["releaseDecision"] != "retain-control"
        or comparison["control"]["checkpointSha256"]
        != control_input["checkpointSha256"]
        or comparison["bestTreatment"]["checkpointSha256"]
        != treatment_input["checkpointSha256"]
    ):
        raise CapturableDatasetError(
            "prior comparison does not bind the frozen candidates"
        )
    inputs = {
        "control": control_input,
        "priorComparison": {
            "file": resolved_comparison.name,
            "releaseDecision": comparison["releaseDecision"],
            "sha256": comparison_sha256,
        },
        "treatment": treatment_input,
        "validation": validation_identity,
    }
    return (
        control_model,
        control_metadata,
        control_input,
        treatment_model,
        treatment_metadata,
        treatment_input,
        tuple(rows),
        inputs,
        comparison,
    )


def run_blend_validation(
    control_selection_path: Path,
    treatment_selection_path: Path,
    validation_path: Path,
    prior_comparison_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Evaluate the frozen grid without accepting or opening a test path."""

    if output_path.exists():
        raise FileExistsError("blend validation output already exists")
    (
        control_model,
        control_metadata,
        _control_input,
        treatment_model,
        treatment_metadata,
        _treatment_input,
        rows,
        inputs,
        _comparison,
    ) = _load_frozen_inputs(
        control_selection_path,
        treatment_selection_path,
        validation_path,
        prior_comparison_path,
    )

    tensors = tensorize(rows)
    control_predictions = component_predictions(
        control_model,
        control_metadata,
        rows,
        tensors,
    )
    treatment_predictions = component_predictions(
        treatment_model,
        treatment_metadata,
        rows,
        tensors,
    )
    control_metrics = evaluate_predictions(rows, control_predictions)
    treatment_metrics = evaluate_predictions(rows, treatment_predictions)
    if (
        control_metrics != control_metadata["validation"]
        or treatment_metrics != treatment_metadata["validation"]
    ):
        raise CapturableDatasetError(
            "checkpoint predictions do not reproduce selected validation"
        )

    candidates = []
    for weight in BLEND_WEIGHTS:
        predictions = blend_components(
            rows,
            control_predictions,
            treatment_predictions,
            weight,
        )
        candidates.append(
            {
                "metrics": evaluate_predictions(rows, predictions),
                "predictionsSha256": predictions.sha256,
                "weight": weight,
            }
        )
    selected = max(candidates, key=candidate_order)
    primary_confirmed = (
        performance_order(selected["metrics"])
        > performance_order(control_metrics)
    )
    reliability_checks = blend_reliability_checks(
        control_metrics,
        selected["metrics"],
        primary_confirmed,
    )
    release_decision = (
        "promote-blend"
        if all(reliability_checks.values())
        else "retain-control"
    )
    artifact = {
        "candidates": candidates,
        "control": {
            "metrics": control_metrics,
            "predictionsSha256": control_predictions.sha256,
            "weight": 0.0,
        },
        "execution": _execution_identity(),
        "format": CAPTURABLE_BLEND_FORMAT,
        "inputs": inputs,
        "primaryDecision": (
            "confirm-blend" if primary_confirmed else "reject-blend"
        ),
        "protocol": {
            "commit": PROTOCOL_COMMIT,
            "file": PROTOCOL_FILE,
            "sha256": PROTOCOL_SHA256,
        },
        "releaseDecision": release_decision,
        "reliabilityChecks": reliability_checks,
        "sealedTestStatus": "unopened",
        "selected": selected,
        "selectionMetric": SELECTION_METRIC,
        "treatment": {
            "metrics": treatment_metrics,
            "predictionsSha256": treatment_predictions.sha256,
            "weight": 1.0,
        },
        "version": CAPTURABLE_BLEND_VERSION,
        "weightGrid": list(BLEND_WEIGHTS),
    }
    payload = _canonical_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    selected_hybrid = selected["metrics"]["hybrid"]
    control_hybrid = control_metrics["hybrid"]
    return {
        "artifactPath": str(output_path),
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "controlGameNormalizedTop1": control_hybrid[
            "game_normalized_top_1_accuracy"
        ],
        "releaseDecision": release_decision,
        "selectedGameNormalizedTop1": selected_hybrid[
            "game_normalized_top_1_accuracy"
        ],
        "selectedWeight": selected["weight"],
    }


def load_blend_validation(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load a canonical blend decision and recompute its selection decision."""

    artifact, sha256 = _selection_report(path)
    expected_keys = {
        "candidates",
        "control",
        "execution",
        "format",
        "inputs",
        "primaryDecision",
        "protocol",
        "releaseDecision",
        "reliabilityChecks",
        "sealedTestStatus",
        "selected",
        "selectionMetric",
        "treatment",
        "version",
        "weightGrid",
    }
    candidates = artifact.get("candidates")
    control = artifact.get("control")
    treatment = artifact.get("treatment")
    execution = artifact.get("execution")
    expected_inputs = {
        "control": dict(_EXPECTED_INPUTS["control"]),
        "priorComparison": {
            **_EXPECTED_INPUTS["priorComparison"],
            "releaseDecision": "retain-control",
        },
        "treatment": dict(_EXPECTED_INPUTS["treatment"]),
        "validation": dict(_EXPECTED_INPUTS["validation"]),
    }
    if (
        set(artifact) != expected_keys
        or artifact.get("format") != CAPTURABLE_BLEND_FORMAT
        or artifact.get("version") != CAPTURABLE_BLEND_VERSION
        or artifact.get("protocol")
        != {
            "commit": PROTOCOL_COMMIT,
            "file": PROTOCOL_FILE,
            "sha256": PROTOCOL_SHA256,
        }
        or artifact.get("weightGrid") != list(BLEND_WEIGHTS)
        or artifact.get("selectionMetric") != SELECTION_METRIC
        or artifact.get("inputs") != expected_inputs
        or artifact.get("sealedTestStatus") != "unopened"
        or not isinstance(execution, Mapping)
        or set(execution)
        != {"cleanWorktree", "repository", "revision"}
        or execution.get("cleanWorktree") is not True
        or execution.get("repository") != "DrawbackGuesser"
        or not isinstance(execution.get("revision"), str)
        or len(execution["revision"]) != 40
        or any(
            token not in "0123456789abcdef"
            for token in execution["revision"]
        )
        or not isinstance(candidates, list)
        or len(candidates) != len(BLEND_WEIGHTS)
        or not isinstance(control, Mapping)
        or not isinstance(treatment, Mapping)
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible blend validation"
        )
    if [candidate.get("weight") for candidate in candidates] != list(
        BLEND_WEIGHTS
    ):
        raise CapturableDatasetError(
            f"{path.name} blend grid is not ordered and complete"
        )
    entries = [control, *candidates, treatment]
    if any(
        set(entry) != {"metrics", "predictionsSha256", "weight"}
        or not isinstance(entry.get("metrics"), Mapping)
        or not _is_sha256(entry.get("predictionsSha256"))
        for entry in entries
    ):
        raise CapturableDatasetError(
            f"{path.name} blend candidate shape is invalid"
        )
    if control.get("weight") != 0.0 or treatment.get("weight") != 1.0:
        raise CapturableDatasetError(
            f"{path.name} component weights are invalid"
        )
    selected = max(candidates, key=candidate_order)
    if artifact.get("selected") != selected:
        raise CapturableDatasetError(
            f"{path.name} selected blend is inconsistent"
        )
    control_metrics = control.get("metrics")
    selected_metrics = selected.get("metrics")
    if not isinstance(control_metrics, Mapping) or not isinstance(
        selected_metrics,
        Mapping,
    ):
        raise CapturableDatasetError(
            f"{path.name} blend metrics are invalid"
        )
    primary_confirmed = (
        performance_order(selected_metrics)
        > performance_order(control_metrics)
    )
    reliability_checks = blend_reliability_checks(
        control_metrics,
        selected_metrics,
        primary_confirmed,
    )
    decision = (
        "promote-blend"
        if all(reliability_checks.values())
        else "retain-control"
    )
    if (
        artifact.get("primaryDecision")
        != ("confirm-blend" if primary_confirmed else "reject-blend")
        or artifact.get("reliabilityChecks") != reliability_checks
        or artifact.get("releaseDecision") != decision
        or treatment != candidates[-1]
    ):
        raise CapturableDatasetError(
            f"{path.name} blend decision is inconsistent"
        )
    return artifact, sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the preregistered capturable 25-label convex blend "
            "on selection validation only."
        )
    )
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--treatment-selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = run_blend_validation(
        options.control_selection,
        options.treatment_selection,
        options.validation,
        options.prior_comparison,
        options.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
