"""Stable, bounded tokenization for public SAN move history."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal, Mapping, Sequence

from .records import FeatureRecord


PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
TOKENIZER_VERSION = 1


@dataclass(frozen=True)
class SanTokenizer:
    """An exact-token SAN vocabulary fitted only from training examples."""

    vocabulary: tuple[str, ...]
    max_history: int

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        if len(self.vocabulary) < 2:
            raise ValueError("SAN vocabulary must include reserved tokens")
        if self.vocabulary[:2] != (PAD_TOKEN, UNKNOWN_TOKEN):
            raise ValueError("SAN vocabulary has invalid reserved-token order")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("SAN vocabulary tokens must be unique")

    @classmethod
    def fit(
        cls,
        histories: Iterable[Sequence[str]],
        *,
        max_history: int,
    ) -> "SanTokenizer":
        tokens = sorted({token for history in histories for token in history})
        return cls((PAD_TOKEN, UNKNOWN_TOKEN, *tokens), max_history)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "SanTokenizer":
        if (
            metadata.get("kind") != "exact-san-token"
            or metadata.get("padding") != "right"
            or metadata.get("truncation") != "keep-most-recent"
        ):
            raise ValueError("incompatible SAN tokenizer semantics")
        if metadata.get("version") != TOKENIZER_VERSION:
            raise ValueError("unsupported SAN tokenizer version")
        vocabulary = metadata.get("vocabulary")
        max_history = metadata.get("max_history")
        if (
            not isinstance(vocabulary, (list, tuple))
            or any(not isinstance(token, str) for token in vocabulary)
            or not isinstance(max_history, int)
            or isinstance(max_history, bool)
        ):
            raise ValueError("invalid SAN tokenizer metadata")
        return cls(tuple(vocabulary), max_history)

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "exact-san-token",
            "version": TOKENIZER_VERSION,
            "vocabulary": list(self.vocabulary),
            "max_history": self.max_history,
            "padding": "right",
            "truncation": "keep-most-recent",
        }

    def encode(self, history: Sequence[str]) -> tuple[tuple[int, ...], int]:
        token_index = {token: index for index, token in enumerate(self.vocabulary)}
        bounded = history[-self.max_history :]
        encoded = [token_index.get(token, 1) for token in bounded]
        length = len(encoded)
        padded = encoded + [0] * (self.max_history - len(encoded))
        return tuple(padded), length

    def encode_batch(
        self, histories: Iterable[Sequence[str]]
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(self.encode(history)[0] for history in histories)


OBSERVATION_TOKENIZER_VERSION = 2
OBSERVATION_TOKENIZER_KIND = "public-sequence-observation-token"
UNKNOWN_CURRENT_MOVE_TOKEN = "<unk-current-move>"
MASKED_CURRENT_MOVE_TOKEN = "<current-move-masked>"
CURRENT_MOVE_TOKEN_PREFIX = "<move:"
_CURRENT_MOVE_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$")
MAX_OBSERVATION_SAN_TOKEN_LENGTH = 32
SequenceObservationMode = Literal["masked-current-v2", "exact-current-v2"]
SEQUENCE_OBSERVATION_MODES: tuple[SequenceObservationMode, ...] = (
    "masked-current-v2",
    "exact-current-v2",
)


def current_move_token(move_uci: str) -> str:
    """Namespace one exact, public current move for sequence encoding."""

    if not isinstance(move_uci, str) or _CURRENT_MOVE_PATTERN.fullmatch(move_uci) is None:
        raise ValueError("current move must be canonical UCI")
    origin = move_uci[:2]
    destination = move_uci[2:4]
    if origin == destination or (len(move_uci) == 5 and destination[1] not in {"1", "8"}):
        raise ValueError("current move must be canonical UCI")
    return f"{CURRENT_MOVE_TOKEN_PREFIX}{move_uci}>"


@dataclass(frozen=True)
class PublicSequenceObservation:
    """The complete public input accepted by the v2 sequence tokenizer."""

    prior_san: tuple[str, ...]
    current_move_uci: str

    def __post_init__(self) -> None:
        if type(self.prior_san) is not tuple:
            raise TypeError("prior_san must be a tuple of public SAN tokens")
        prior = self.prior_san
        if any(
            not isinstance(token, str)
            or not token
            or len(token) > MAX_OBSERVATION_SAN_TOKEN_LENGTH
            or token
            in {
                PAD_TOKEN,
                UNKNOWN_TOKEN,
                UNKNOWN_CURRENT_MOVE_TOKEN,
                MASKED_CURRENT_MOVE_TOKEN,
            }
            or token.startswith(CURRENT_MOVE_TOKEN_PREFIX)
            or "<" in token
            or ">" in token
            for token in prior
        ):
            raise ValueError("prior SAN contains an invalid or reserved token")
        current_move_token(self.current_move_uci)
        object.__setattr__(self, "prior_san", prior)


def public_sequence_observation(
    features: FeatureRecord,
) -> PublicSequenceObservation:
    """Build the v2 sequence input from the public feature boundary only."""

    if not isinstance(features, FeatureRecord):
        raise TypeError("features must be a FeatureRecord")
    return PublicSequenceObservation(features.history_san, features.move)


def observation_tokens(
    observation: PublicSequenceObservation,
    *,
    mask_current: bool = False,
) -> tuple[str, ...]:
    """Return prior SAN followed by exactly one current-move token."""

    if not isinstance(observation, PublicSequenceObservation):
        raise TypeError("observation must be PublicSequenceObservation")
    if not isinstance(mask_current, bool):
        raise TypeError("mask_current must be boolean")
    current = (
        MASKED_CURRENT_MOVE_TOKEN
        if mask_current
        else current_move_token(observation.current_move_uci)
    )
    return (*observation.prior_san, current)


@dataclass(frozen=True)
class ObservationTokenizerV2:
    """Version-2 tokenizer for prior SAN plus one exact current UCI move."""

    vocabulary: tuple[str, ...]
    max_sequence: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_sequence, int)
            or isinstance(self.max_sequence, bool)
            or self.max_sequence <= 0
        ):
            raise ValueError("max_sequence must be positive")
        if self.vocabulary[:4] != (
            PAD_TOKEN,
            UNKNOWN_TOKEN,
            UNKNOWN_CURRENT_MOVE_TOKEN,
            MASKED_CURRENT_MOVE_TOKEN,
        ):
            raise ValueError("observation tokenizer has invalid reserved-token order")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("observation tokenizer tokens must be unique")
        if any(not isinstance(token, str) or not token for token in self.vocabulary):
            raise ValueError("observation tokenizer tokens must be non-empty strings")
        for token in self.vocabulary[4:]:
            if token.startswith(CURRENT_MOVE_TOKEN_PREFIX):
                move_uci = token[len(CURRENT_MOVE_TOKEN_PREFIX) : -1]
                try:
                    expected_token = current_move_token(move_uci)
                except ValueError as error:
                    raise ValueError(
                        "observation tokenizer has invalid current-move token"
                    ) from error
                if token != expected_token:
                    raise ValueError("observation tokenizer has invalid current-move token")
            elif (
                token
                in {
                    PAD_TOKEN,
                    UNKNOWN_TOKEN,
                    UNKNOWN_CURRENT_MOVE_TOKEN,
                    MASKED_CURRENT_MOVE_TOKEN,
                }
                or "<" in token
                or ">" in token
                or len(token) > MAX_OBSERVATION_SAN_TOKEN_LENGTH
            ):
                raise ValueError("observation tokenizer has invalid SAN token")

    @classmethod
    def fit(
        cls,
        observations: Iterable[PublicSequenceObservation],
        *,
        max_sequence: int,
    ) -> "ObservationTokenizerV2":
        tokens: set[str] = set()
        observed = False
        for observation in observations:
            if not isinstance(observation, PublicSequenceObservation):
                raise TypeError("observations must contain PublicSequenceObservation")
            observed = True
            tokens.update(observation.prior_san)
            tokens.add(current_move_token(observation.current_move_uci))
        if not observed:
            raise ValueError("at least one public sequence observation is required")
        return cls(
            (
                PAD_TOKEN,
                UNKNOWN_TOKEN,
                UNKNOWN_CURRENT_MOVE_TOKEN,
                MASKED_CURRENT_MOVE_TOKEN,
                *sorted(tokens),
            ),
            max_sequence,
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> "ObservationTokenizerV2":
        if (
            metadata.get("kind") != OBSERVATION_TOKENIZER_KIND
            or metadata.get("version") != OBSERVATION_TOKENIZER_VERSION
            or metadata.get("padding") != "right"
            or metadata.get("truncation") != "keep-most-recent"
            or metadata.get("current_move") != "required-final-namespaced-uci"
        ):
            raise ValueError("incompatible observation tokenizer semantics")
        vocabulary = metadata.get("vocabulary")
        max_sequence = metadata.get("max_sequence")
        if (
            not isinstance(vocabulary, (list, tuple))
            or any(not isinstance(token, str) for token in vocabulary)
            or not isinstance(max_sequence, int)
            or isinstance(max_sequence, bool)
        ):
            raise ValueError("invalid observation tokenizer metadata")
        return cls(tuple(vocabulary), max_sequence)

    def metadata(self) -> dict[str, object]:
        return {
            "kind": OBSERVATION_TOKENIZER_KIND,
            "version": OBSERVATION_TOKENIZER_VERSION,
            "vocabulary": list(self.vocabulary),
            "max_sequence": self.max_sequence,
            "padding": "right",
            "truncation": "keep-most-recent",
            "current_move": "required-final-namespaced-uci",
        }

    def encode(
        self,
        observation: PublicSequenceObservation,
        *,
        mask_current: bool = False,
    ) -> tuple[tuple[int, ...], int]:
        token_index = {token: index for index, token in enumerate(self.vocabulary)}
        bounded = observation_tokens(
            observation,
            mask_current=mask_current,
        )[-self.max_sequence :]
        encoded = [
            token_index.get(
                token,
                2 if token.startswith(CURRENT_MOVE_TOKEN_PREFIX) else 1,
            )
            for token in bounded
        ]
        length = len(encoded)
        padded = encoded + [0] * (self.max_sequence - length)
        return tuple(padded), length


SequenceTokenizer = SanTokenizer | ObservationTokenizerV2


def encode_public_sequence(
    tokenizer: SequenceTokenizer,
    features: FeatureRecord,
    sequence_observation_mode: SequenceObservationMode | None,
) -> tuple[tuple[int, ...], int]:
    """Encode one public row under the checkpoint's explicit sequence contract."""

    if isinstance(tokenizer, ObservationTokenizerV2):
        if sequence_observation_mode not in SEQUENCE_OBSERVATION_MODES:
            raise ValueError(
                "observation tokenizer requires an explicit observation mode"
            )
        return tokenizer.encode(
            public_sequence_observation(features),
            mask_current=sequence_observation_mode == "masked-current-v2",
        )
    if sequence_observation_mode is not None:
        raise ValueError("legacy SAN tokenizer cannot encode a current-move mode")
    return tokenizer.encode(features.history_san)
