# Mirage

Mirage is a Go implementation of a TurboQuant-native image codec and `.mrg`
bitstream. It owns the host-side codec substrate, browser decoder path, image
I/O, arithmetic coding, latent payload coding, and Manta integration used by the
learned codec entry points.

## What Is Implemented

- `.mrg` fixed 72-byte headers, payload slicing, and CRC-32 validation.
- Deterministic 32-bit arithmetic coding shared by the Go encoder and WASM
  decoder.
- TurboQuant categorical and bit-plane latent payload coding for coordinate
  streams plus q_norm streams.
- Per-symbol learned entropy model hooks for Manta logits, probabilities, and
  log-normal q_norm parameters.
- Log-space q_norm quantization.
- Channel-first RGB helpers with PNG, JPEG, and PPM support.
- TurboQuant latent adapter that quantizes one spatial position as one vector
  over latent channels.
- `mirage` CLI commands for encode, decode, metadata inspection, evaluation,
  Manta artifact initialization, training smoke tests, Kodak training, and
  CompressAI baseline fetching.
- Manta artifact path for exporting the Mirage Image `.mll` operator surface.
- Learned Manta checkpoint path for `mirage encode`, `mirage decode`, and
  `mirage eval` through `-manta-module` and `-manta-weights`.

Mirage depends on public Manta and TurboQuant module versions. The standalone
workspace has no local module replacements.

## Current Baseline

The 2000-step Adam/cosine reference recipe on 10 Kodak center crops reaches
`22.2543 dB` at `0.3355 bpp` for lambda `0.01`. The same recipe is stable for
lambda `0.001`.

On the all-24 Kodak eval set, the center-crop checkpoint reaches `21.7257 dB`
at `0.3398 bpp`. A one-random-crop transfer run is stable at `21.5218 dB` and
`0.3481 bpp`.

An 8-bit/32-latent/16-hyper capacity smoke reaches `21.8730 dB` by step 1000
at `1.2173 bpp`.

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
go run ./cmd/mirage train-manta-kodak -dir kodak -max-images 5 -steps 200 -crop 256 -lambdas 0.001,0.01,0.1 -optimizer adam -lr 0.001 -out-dir runs/kodak-reference
go run ./cmd/mirage train-manta-kodak -dir kodak -max-images 10 -steps 2000 -crop 256 -lambdas 0.01 -bits 4 -latent-channels 16 -hyper-channels 8 -optimizer adam -lr 0.001 -lr-schedule cosine -lr-final 0.000001 -clip 1 -checkpoint-every 100 -out-dir runs/kodak-short
go run ./cmd/mirage eval-manta-kodak -dir kodak -run-dir runs/kodak-reference -out-dir runs/kodak-eval
go run ./cmd/mirage fetch-compressai-baseline -out-dir baselines/compressai -architectures bmshj2018-factorized,bmshj2018-hyperprior -qualities 1,2,3,4,5,6,7,8
go run ./cmd/mirage encode -in input.png -out learned.mrg -manta-module runs/kodak-reference/mirage_v1_lambda_0p001.mll -manta-weights runs/kodak-reference/mirage_v1_lambda_0p001.weights.mll
go run ./cmd/mirage eval -source input.png -mrg learned.mrg -manta-module runs/kodak-reference/mirage_v1_lambda_0p001.mll -manta-weights runs/kodak-reference/mirage_v1_lambda_0p001.weights.mll
```

Build the browser decoder assets:

```bash
./scripts/build_web.sh
```

## License

Apache-2.0. See `LICENSE`.
