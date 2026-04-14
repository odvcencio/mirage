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
  `ExecuteAutograd` with clipped SGD or Adam
- `mirage train-manta-kodak` walks a real image directory, selects a bounded
  subset, runs a lambda sweep through the same reference autograd path, and
  writes one `.mll` module plus one `.weights.mll` checkpoint per lambda
- `train-manta-kodak` supports `-optimizer adam`, `-crop-mode random`,
  `-random-crops-per-image`, `-crop-seed`, and `-resume` so longer reference
  runs can train on many deterministic random crops and continue from an
  existing `.weights.mll` checkpoint
- `mirage encode`, `mirage decode`, and `mirage eval` accept
  `-manta-module` plus `-manta-weights` to run the learned Manta deployment
  path: `analyze`, arithmetic-coded `c_z`, hyperprior synthesis,
  conditional `c_coords` / `c_norms`, and `synthesize_image`

Remaining work to reach the publishable Balle 2018 result:

- CUDA forward/backward kernels promoted from host-reference execution
- WebGPU device kernels promoted from host-reference execution
- dataset batching and metric sweeps beyond the current Kodak subset reference
  gate
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
lambda, so the rate signal now reaches the analysis path. At this point
calibration was still not solved. Final MSE remained effectively identical
across the three lambda points, and the lowest lambda still produced the lowest
final rate. A 10x clip relaxation at lambda 0.1 diverged to `+Inf`; `clip=1`
stayed finite but increased rate to `30536.723`; `clip=5` with a 10x lower
learning rate stayed finite but ended at MSE `0.063768` and rate `25066.537`.
Those results triggered the longer baseline and loss-unit audit below.

The next baseline is a single middle-lambda run over all 24 Kodak images:

```bash
mirage train-manta-kodak \
  -dir /tmp/mirage-kodak-all-24 \
  -max-images 24 \
  -steps 5000 \
  -crop 256 \
  -lambdas 0.01 \
  -bits 4 \
  -latent-channels 16 \
  -hyper-channels 8 \
  -out-dir /tmp/mirage-kodak-runs/long-baseline
```

This intentionally keeps the current center-crop regime so it answers one
question cleanly: whether the existing 4-bit x 16-latent x 8-hyper model reaches
a meaningful operating point when trained for long enough.

That baseline completed in `1h9m14s` and did not reach the target regime:
lambda `0.01` ended at MSE `0.046059` (`13.37 dB` proxy PSNR) and rate
`22892.527`. The loss decomposition showed the structural unit mismatch:
`loss = mse + lambda * rate` used summed image bits, so even lambda `0.01`
made the rate term roughly 99.9% of the loss.

The current Manta training graph fixes the first-order calibration issues:

- `rate_distortion_loss` now accepts `rate_scale`, and Mirage v1 modules set it
  to `1 / (image_width * image_height)`, so the loss combines MSE with bpp
  rather than summed image bits.
- explicit lambda `0` is preserved for pure-MSE controls instead of being
  normalized to the default lambda.
- `train_step` still emits hard TurboQuant codes for the entropy model, but the
  reconstruction branch synthesizes from continuous `y`; deployment entrypoints
  continue to use hard `analyze` / `synthesize_*` paths.
- the deterministic reference initializer no longer applies the extra `0.25`
  weight-scale shrink, which was starving the encoder of reconstruction
  gradient.
- the reference trainer supports `-optimizer adam` for calibration runs.

Pure-MSE confirmation after those fixes:

```bash
mirage train-manta-kodak \
  -dir /tmp/mirage-kodak-all-24 \
  -max-images 10 \
  -steps 1000 \
  -crop 256 \
  -lambdas 0 \
  -bits 4 \
  -latent-channels 16 \
  -hyper-channels 8 \
  -optimizer adam \
  -lr 0.001 \
  -out-dir /tmp/mirage-kodak-runs/pure-mse-adam
```

The run ended at MSE `0.007756` (`21.10 dB`) in `14m29s`. A hard `.mrg`
round trip from that checkpoint on the 256x256 center crop of `kodim01.png`
measured MSE `0.009858`, PSNR `20.06 dB`, `0.4336 bpp`, and `3552` container
bytes. This confirms the network can reconstruct through the real deployment
path.

The follow-up Adam sweep used the bpp-normalized RD loss:

```bash
mirage train-manta-kodak \
  -dir /tmp/mirage-kodak-all-24 \
  -max-images 10 \
  -steps 1000 \
  -crop 256 \
  -lambdas 0.001,0.01,0.1 \
  -bits 4 \
  -latent-channels 16 \
  -hyper-channels 8 \
  -optimizer adam \
  -lr 0.001 \
  -clip 1 \
  -out-dir /tmp/mirage-kodak-runs/adam-bpp-sweep
```

Training endpoints:

| lambda | loss | MSE | rate | analysis grad | duration |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.191980 -> 0.008244 | 0.191589 -> 0.007936 | 25626.719 -> 20175.797 | 0.032612 -> 0.084052 | 14m42.5s |
| 0.01 | 0.195500 -> 0.011243 | 0.191589 -> 0.008280 | 25626.719 -> 19419.861 | 0.032810 -> 0.129887 | 14m25.4s |
| 0.1 | 0.230693 -> 0.047937 | 0.191589 -> 0.015155 | 25626.719 -> 21483.936 | 0.035037 -> 0.196333 | 14m19.0s |

That is the first real low-vs-mid lambda rate-distortion signal: lambda `0.01`
buys a lower real rate than lambda `0.001` with only a small distortion cost.
Lambda `0.1` is too aggressive at this optimizer/capacity point and degrades
both quality and final payload rate.

## Evaluation and Baselines

The reference infrastructure now includes a checkpoint evaluation harness that
turns trained Manta weights into real `.mrg` artifacts and measures decoded
container bpp plus PSNR across an image set:

```bash
mirage eval-manta-kodak \
  -dir /tmp/mirage-kodak-all-24 \
  -max-images 10 \
  -run-dir /tmp/mirage-kodak-runs/adam-bpp-sweep \
  -out-dir /tmp/mirage-kodak-runs/adam-bpp-eval
```

Measured `.mrg` deployment results from the 10-image sweep:

| lambda | avg MSE | avg PSNR | avg bpp | avg container bytes |
|---:|---:|---:|---:|---:|
| 0.001 | 0.00796491 | 21.8310 | 0.3498 | 2865.2 |
| 0.01 | 0.00839294 | 21.6181 | 0.3382 | 2770.8 |
| 0.1 | 0.01466716 | 18.5756 | 0.3697 | 3028.6 |

The evaluation harness writes one `.mrg` per model/image pair plus
`eval_summary.json`, so later runs can compare training telemetry against actual
bitstream artifacts instead of proxy losses.

CompressAI baseline checkpoint download is also wired for the Balle 2018
factorized-prior MSE models:

```bash
mirage fetch-compressai-baseline \
  -out-dir /tmp/mirage-baselines/compressai \
  -qualities 1,2,3,4,5,6,7,8 \
  -timeout 1h
```

The first download completed all eight official checkpoints and wrote
`/tmp/mirage-baselines/compressai/manifest.json`. The manifest records the
source URL, checkpoint URLs, byte counts, and SHA-256 hashes. Total downloaded
checkpoint size is about `141M`.

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

This ruled out the "training metric is inverted" branch for the pre-normalized
checkpoints. The real `.mrg` rate tracked the training proxy direction: the
low-lambda checkpoint had lower real bpp than the high-lambda checkpoint, while
PSNR was effectively identical. The bpp-normalized Adam sweep and
`eval-manta-kodak` results above supersede this older calibration conclusion,
but the diagnostic remains useful because it proved the deployment measurement
path was not the source of the inversion.

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
