# Mirage

Mirage is the TurboQuant-native image codec surface described in
`docs/specs/2026-04-12-mirage-v1-design.md`.

This workspace currently lands the executable host-side v1 path:

- `.mrg` v1 fixed 72-byte headers, payload slicing, and CRC-32 validation
- a deterministic 32-bit arithmetic coder shared by encoder and WASM decoder
- factorized latent payload coding for TurboQuant categorical and bit-plane
  coordinate streams plus q_norm streams
- per-symbol learned entropy model hooks for Manta logits/probabilities and
  log-normal q_norm parameters
- q_norm log-space norm quantization
- channel-first RGB image helpers with PNG, JPEG, and PPM support
- a TurboQuant latent adapter that quantizes one spatial position as one vector
  over latent channels
- a `mirage` CLI that can encode PNG/JPEG/PPM to `.mrg`, decode `.mrg` to PNG,
  and print stream metadata
- a Manta import path that exports the Mirage Image v1 `.mll` operator surface
- an image-backed Manta reference training smoke that runs the Mirage v1
  rate-distortion graph through Manta `ExecuteAutograd`

The learned-codec pieces from the spec live in Manta. This repo now imports the
sibling Manta module for the v1 artifact builder, while the standalone
host/WASM code keeps owning `.mrg` parsing, arithmetic coding, q_norm, and image
I/O. Mirage depends on public Manta and TurboQuant module versions; there are no
local module replaces in the standalone workspace.

## Quick Start

```bash
go run ./cmd/mirage encode -in input.png -out input.mrg -bits 4 -factorization categorical
go run ./cmd/mirage encode -in input.png -out input.bitplane.mrg -bits 4 -factorization bit-plane
go run ./cmd/mirage info -in input.mrg
go run ./cmd/mirage decode -in input.mrg -out decoded.png
go run ./cmd/mirage eval -source input.png -mrg input.mrg
go run ./cmd/mirage init-manta -out mirage_v1.mll
go run ./cmd/mirage check-manta -in mirage_v1.mll -entry train_step
go run ./cmd/mirage train-manta-smoke -in input.png -steps 24 -crop 16 -bits 2
go run ./cmd/mirage train-manta-kodak -dir kodak -max-images 5 -steps 200 -crop 256 -lambdas 0.001,0.01,0.1 -out-dir runs/kodak-reference
```

Build the browser decoder assets:

```bash
./scripts/build_web.sh
```

## License

Apache-2.0. See `LICENSE`.
