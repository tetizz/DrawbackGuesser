# Third-party notices

DrawbackGuesser depends on open-source packages including React and Vite
(MIT), TypeScript (Apache-2.0), Vitest (MIT), chess.js (BSD-2-Clause),
chessops (GPL-3.0-or-later), python-chess (GPL-3.0-or-later), and PyTorch
(BSD-style). Their own licenses and notices continue to apply; the lockfile
and Python requirements identify the exact released dependency set. A
distributed browser build containing GPL components must comply with their
GPL terms. The MIT license applies only to original DrawbackGuesser source and
does not replace a dependency's license.

The pinned `DrawbackEngine` Git submodule is a separately versioned component.
Its source, license, rule catalog, and third-party notices remain authoritative
for chess legality, drawback execution, evaluator-sidecar validation, and any
bundled engine assets.

Model weights, checkpoints, simulation corpora, real-game evaluation data, and
other generated artifacts are not covered merely by the source-code license.
Each distributed artifact must include its own provenance, dataset permissions,
content hashes, model card, and any applicable upstream terms. No third-party
game data or model checkpoint is included in this source tree.
