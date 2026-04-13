# Mirage Image v1 Implementation Notes

This repository is now the standalone Mirage workspace. The full design spec is
kept at `docs/specs/2026-04-12-mirage-v1-design.md`.

## Executable v1 Slice

The current Go implementation is a functional `.mrg` image codec:

- `mirage encode` reads PNG, JPEG, or PPM and writes `.mrg`
- `mirage decode` reads `.mrg` and writes PNG
- `.mrg` v1 headers match the 72-byte spec layout
- payloads are CRC-protected and arithmetic-coded as factorized latent streams
- TurboQuant encodes one 4x4 RGB patch as one 48-coordinate latent vector
- `c_coords` payloads encode unpacked TurboQuant coordinate symbols with either
  categorical or bit-plane factorization
- `q_norm` stores one log-space norm byte per latent position
- `c_norms` payloads encode q_norm symbols independently from coordinate
  symbols

This is deliberately the host-side executable slice. It proves the bitstream,
shared arithmetic coder, q_norm representation, TurboQuant latent contract, and
CLI path without blocking on Manta's convolution/autograd/WebGPU work.

The arithmetic coder also exposes `EncodeSymbolsWithModels`,
`DecodeSymbolsWithModels`, `CDFsFromProbabilities`, `CDFsFromLogits`, and
`NormCDFsFromLogNormalParams`. `EncodeLatentPayloadsWithModels` accepts per-symbol
CDF sequences for categorical coordinates, bit-plane coordinates, and q_norm
symbols. Those are the hard v1 host hooks for Manta-produced probability
tensors: the GPU path owns logits/probabilities and log-normal norm parameters,
while Mirage owns deterministic range coding.

## Manta Surface Status

The sibling Manta workspace now has the Mirage v1 operator surface in artifact
schema and host-reference runtime form:

- MIR ops: `conv2d`, `conv2d_transpose`, `gdn`, `igdn`,
  `turboquant_encode`, `turboquant_decode`, `cross_entropy_factorized`,
  `mse_loss`, and `ms_ssim_loss`
- 2D schedule hints: `tile_2d`, `subgroup_2d`, and `halo`
- runtime host-reference implementations for all of the above
- `manta init-mirage` and `manta demo mirage_v1` for a runnable Mirage Image v1
  smoke artifact
- Mirage imports Manta through `github.com/odvcencio/manta` and exposes
  `mirage.MantaModule`, `mirage.EncodeMantaMLL`, and `mirage init-manta` using a
  public Manta pseudo-version with the Mirage surface
- `mirage check-manta` loads an emitted `.mll` through the Manta runtime's
  WebGPU backend surface and executes an entrypoint with deterministic stub
  tensors, keeping the Mirage repo's Manta export path self-checking
- `mirage train-manta-smoke` decodes one or more PNG/JPEG/PPM images,
  center-crops them to the Manta v1 input shape, initializes deterministic
  trainable weights, and runs the `train_step` graph through Manta
  `ExecuteAutograd` with clipped SGD
- `mirage train-manta-kodak` walks a real image directory, selects a bounded
  subset, runs a lambda sweep through the same reference autograd path, and
  writes one `.mll` module plus one `.weights.mll` checkpoint per lambda
- `mirage encode`, `mirage decode`, and `mirage eval` accept
  `-manta-module` plus `-manta-weights` to run the learned Manta deployment
  path: `analyze`, arithmetic-coded `c_z`, hyperprior synthesis,
  conditional `c_coords` / `c_norms`, and `synthesize_image`

Remaining work to reach the publishable Balle 2018 result:

- CUDA forward/backward kernels promoted from host-reference execution
- WebGPU device kernels promoted from host-reference execution
- dataset loading, batching, optimizer calibration, and metric sweeps beyond the
  current Kodak subset reference gate
- longer reference and accelerated training runs that produce publishable
  rate-distortion checkpoints

The implementation boundary is intentional: the `.mrg` file substrate and host
codec APIs live here, while the learned analysis/synthesis network belongs in
Manta.

## Reference Training Gate

Before writing more backward kernels, v1 now has a real-data reference gate over
the Kodak Lossless True Color Image Suite. The first gate run used `kodim01.png`
through `kodim05.png`, center-cropped each image to 256x256, trained the
reference Manta graph, and wrote Manta module plus weight checkpoints for each
lambda point.

The first 2-bit x 4-channel x 4-hyper run exposed a bug: the rate term reached
entropy logits but did not propagate a surrogate gradient into TurboQuant code
indices, so `g_a` was effectively trained only by distortion. Manta fixed that
by adding finite-difference code-index gradients to `cross_entropy_factorized`
and q_norm STE gradients back to the latent vector magnitude. The Kodak command
now records raw gradient norms by graph region so this stays visible.

Command:

```bash
go run ./cmd/mirage train-manta-kodak \
  -dir /tmp/mirage-kodak \
  -max-images 5 \
  -steps 500 \
  -crop 256 \
  -lambdas 0.001,0.01,0.1 \
  -bits 4 \
  -latent-channels 16 \
  -hyper-channels 8 \
  -out-dir /tmp/mirage-kodak-runs/capacity
```

Observed after the rate-gradient fix on 2026-04-13:

| lambda | loss | MSE | rate | analysis grad | duration |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 28.200409 -> 20.649328 | 0.158276 -> 0.031796 | 28042.131 -> 20617.531 | 26.966 -> 8.502 | 7m05.8s |
| 0.01 | 280.579620 -> 275.010864 | 0.158276 -> 0.031796 | 28042.131 -> 27497.906 | 269.655 -> 79.649 | 7m05.3s |
| 0.1 | 2804.371826 -> 2486.961182 | 0.158276 -> 0.031796 | 28042.131 -> 24869.295 | 2696.551 -> 1037.287 | 7m05.4s |

The rate-gradient fix worked: the first-step analysis gradient norm scales with
lambda, so the rate signal now reaches the analysis path. Calibration is still
not solved. Final MSE remains effectively identical across the three lambda
points, and the lowest lambda still produces the lowest final rate. A 10x clip
relaxation at lambda 0.1 diverged to `+Inf`; `clip=1` stayed finite but increased
rate to `30536.723`; `clip=5` with a 10x lower learning rate stayed finite but
ended at MSE `0.063768` and rate `25066.537`. The next blocker is optimizer /
surrogate calibration for high-lambda rate pressure, not CUDA backward kernels.

## Deployment Round Trip

The first learned-codec deployment round trip now loads checkpointed
`.weights.mll` files, runs Manta `analyze`, converts Manta logits and log-normal
norm parameters into per-symbol arithmetic CDFs, writes real `.mrg` payloads,
decodes through `synthesize_hyperprior` and `synthesize_image`, and measures
container bpp plus PSNR.

The diagnostic also fixed the `train-manta-kodak -out-dir` artifact writer: it
now emits modules after CLI shape normalization, so `-crop 256` produces a
256x256 `.mll` rather than the 16x16 default artifact.

Command shape used for the 2026-04-13 checkpoint diagnostic:

```bash
mirage encode \
  -in /tmp/mirage-kodak/kodim01.png \
  -out /tmp/mirage-roundtrip/lambda-0p001.mrg \
  -manta-module /tmp/mirage-roundtrip/modules/lambda-0p001-256.mll \
  -manta-weights /tmp/mirage-kodak-runs/capacity/lambda-0p001/mirage_v1_lambda_0p001.weights.mll

mirage eval \
  -source /tmp/mirage-kodak/kodim01.png \
  -mrg /tmp/mirage-roundtrip/lambda-0p001.mrg \
  -manta-module /tmp/mirage-roundtrip/modules/lambda-0p001-256.mll \
  -manta-weights /tmp/mirage-kodak-runs/capacity/lambda-0p001/mirage_v1_lambda_0p001.weights.mll
```

Real `.mrg` measurements on the 256x256 center crop of `kodim01.png`:

| lambda | training rate end | container bytes | bpp | PSNR |
|---:|---:|---:|---:|---:|
| 0.001 | 20617.531 | 2965 | 0.3619 | 14.9001 |
| 0.01 | 27497.906 | 3780 | 0.4614 | 14.9008 |
| 0.1 | 24869.295 | 3447 | 0.4208 | 14.9009 |

This rules out the "training metric is inverted" branch. The real `.mrg` rate
tracks the training proxy direction: the low-lambda checkpoint has lower real
bpp than the high-lambda checkpoint, while PSNR is effectively identical. The
remaining blocker is therefore training dynamics / surrogate bias causing
high-lambda optimization to produce a worse entropy model, not a deployment
measurement mismatch.

## Visual System

Territory: Swiss Precision. The browser decoder uses a direct tool layout with
clear hierarchy, sharp alignment, and no decorative blobs or marketing hero.

Typography:

- Display: Space Grotesk, 600
- Body: Work Sans, 400 and 600
- Mono: IBM Plex Mono, 500
- Type scale: Minor Third, 1.2

Color architecture:

- Dominant 60%: `#FFFFFF`
- Secondary 30%: `#F2F4F1`
- Accent 10%: `#0B7A5A`
- Text primary: `#101312` on dominant, contrast 18.4:1, WCAG AAA
- Text secondary: `#3D4742` on dominant, contrast 9.0:1, WCAG AAA
- Text muted: `#69746F` on dominant, contrast 4.9:1, WCAG AA

Motion:

- Philosophy: Minimal
- Duration fast: 150ms
- Duration normal: 200ms
- ease-out: `cubic-bezier(0.16, 1, 0.3, 1)`
- ease-spring: `cubic-bezier(0.34, 1.56, 0.64, 1)`

Spacing:

- Base: 8px
- xs: `0.5rem`
- sm: `0.75rem`
- md: `1rem`
- lg: `1.5rem`
- xl: `2rem`
- 2xl: `3rem`
- 3xl: `4rem`

CSS custom properties:

```css
:root {
  --font-display: "Space Grotesk", "Work Sans", sans-serif;
  --font-body: "Work Sans", sans-serif;
  --font-mono: "IBM Plex Mono", monospace;
  --type-xs: 0.694rem;
  --type-sm: 0.833rem;
  --type-md: 1rem;
  --type-lg: 1.2rem;
  --type-xl: 1.44rem;
  --type-2xl: 1.728rem;
  --type-3xl: 2.074rem;
  --color-dominant: #ffffff;
  --color-secondary: #f2f4f1;
  --color-accent: #0b7a5a;
  --color-text-primary: #101312;
  --color-text-secondary: #3d4742;
  --color-text-muted: #69746f;
  --color-border: #cdd5cf;
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --space-xs: 0.5rem;
  --space-sm: 0.75rem;
  --space-md: 1rem;
  --space-lg: 1.5rem;
  --space-xl: 2rem;
  --space-2xl: 3rem;
  --space-3xl: 4rem;
}
```
