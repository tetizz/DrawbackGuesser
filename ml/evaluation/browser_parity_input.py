"""CLI for canonical public-PGN browser parity input production."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .browser_parity import build_public_parity_input
from .ensemble_calibration import ContentAddressedFile
from .release_selection_bundle import ContentAddressedJson


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build authenticated Python expectations for public PGNs."
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("browser_artifact", type=Path)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--browser-artifact-sha256", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    build_public_parity_input(
        fixture=ContentAddressedFile(
            arguments.fixture, arguments.fixture_sha256
        ),
        ensemble_release=ContentAddressedJson(
            arguments.ensemble_release, arguments.ensemble_sha256
        ),
        calibration=ContentAddressedFile(
            arguments.calibration, arguments.calibration_sha256
        ),
        browser_artifact=ContentAddressedFile(
            arguments.browser_artifact, arguments.browser_artifact_sha256
        ),
        repository=arguments.repository.resolve(),
        output=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
