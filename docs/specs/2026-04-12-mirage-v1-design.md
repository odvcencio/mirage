# Mirage v1 — Design Spec

**Status:** approved design, implementation not yet started
**Date:** 2026-04-12
**Codename:** Mirage (working name; may be renamed before public release)
**Scope:** Mirage Image v1 (Ballé 2018 backbone, TurboQuant quantizer, Manta-native, WebGPU browser decoder)
**Author:** odvcencio with Claude

---

## 1. Scope and phase commitments

Mirage is a three-phase project, single throughline: *a neural image codec built natively in Manta, with TurboQuant as a first-class quantizer, where each phase lands both a shippable codec and a reusable Manta surface.*

### Phase 1 — Mirage Image v1 (this spec)

v1 is "done" when the Ballé 2018 analysis and synthesis networks run end-to-end in Manta with TurboQuant as the quantizer (replacing standard additive-uniform-noise scalar quantization), a factorized TurboQuant-codeword entropy model (not a Gaussian hyperprior), trained on a standard corpus (Vimeo-90k frames + CLIC training), with rate-distortion measured on Kodak and CLIC2020 and baselined against CompressAI's reference Ballé 2018. The browser demo is a drop-a-file-on-a-canvas page that decodes a `.mrg` file via the `.mll` WebGPU backend.

**v1 lands in Manta:**

- `conv2d` MIR op with 2D schedule hints
- `conv2d_transpose` for synthesis-net upsampling
- Forward and backward CUDA kernels for both conv ops (hand-written, no cuDNN dependency, matching Manta's pure-Go toolchain rule)
- A promoted WebGPU device kernel for `conv2d` and `conv2d_transpose` — not the current host-fallback stub
- `gdn` and `igdn` as first-class MIR ops with forward and backward kernels
- The TurboQuant quantizer exposed as first-class Manta ops (`turboquant_encode`, `turboquant_decode`) with a straight-through gradient
- A generic autograd path beyond contrastive loss, supporting conv2d, conv2d_transpose, gdn/igdn, the TurboQuant quantizer, matmul, softmax, LayerNorm, pointwise ops, and new loss primitives (cross-entropy against categorical and factorized distributions, MSE, MS-SSIM)
- Image I/O and a rate-distortion loss primitive in the training path

### Phase 2 — Mirage Image v2

Upgrade the v1 codec with residual blocks, simplified attention modules (native matmul + softmax), and the first richer conditional entropy model: a TurboQuant-codeword-aware conditional model that predicts codeword probabilities conditional on neighbors rather than via a hyperprior alone. "Done" when it matches or beats BPG / H.265 intra rate-distortion on Kodak.

**v2 lands in Manta:** residual block primitive, a generalized conditional entropy model interface that retroactively accommodates the v1 model, and any new TurboQuant machinery the conditional model needs.

### Phase 3 — Mirage Image v3

Autoregressive spatial context over TurboQuant codewords, running on the same quantized-sequence decode kernels Manta is building for the hybrid KV cache work. "Done" when it approaches SOTA on Kodak + CLIC2020. The browser decoder uses Manta's autoregressive decode path — the same machinery as LLM decode.

**v3 lands in Manta:** masked `conv2d` and/or native autoregressive iteration over TurboQuant codeword streams, and the conceptual unification "image codec decoder = LLM decoder" at the runtime level — both become decode loops over quantized token streams.

### Phase ordering discipline

v1 blocks v2 blocks v3. Each phase must ship its codec, its Manta surface, its writeup, and its browser demo before the next begins. Parallelizing phases is explicitly rejected: each phase's Manta surface is what the next phase builds on, and divergent work undoes that discipline.

### Final deliverables at the end of the project

- Three versions of Mirage Image, each with its own m31labs writeup
- A permanently stronger Manta: `conv2d` + transposed `conv2d` + residual blocks + TurboQuant dtype with straight-through gradient + conditional entropy interface + masked / autoregressive codeword surface, all first-class and reusable
- A browser demo for each phase
- One research narrative: *TurboQuant is the right quantizer for learned image coding, demonstrated at three increasingly sophisticated architectural regimes, all running in a Go-native ML stack with browser-side WebGPU decoding*

---

## 2. v1 Architecture

Ballé 2018's backbone, with every piece Manta/TurboQuant-native from day one rather than ported-with-quantizer-glue. The decisive architectural choice: v1's entropy model already predicts TurboQuant codeword probabilities directly, not Gaussian parameters over a pre-quantization latent. v1 is already "the Manta/TurboQuant way" at a small scale, and v2/v3 are richer conditional models on the same interface rather than rewrites.

### Components

1. **Analysis net `g_a`.** A conv2d stack in the Ballé 2018 topology: 4 strided conv layers producing a latent tensor `y ∈ ℝ^{C × H/16 × W/16}`. Written as a Manta pipeline of `conv2d` + GDN + pointwise nonlinearities.
2. **TurboQuant forward.** `y` is mapped to TurboQuant codeword indices `c_coords` plus per-position norms `c_norms`. Exposed as first-class Manta ops, not a host-side call.
3. **Hyperprior analysis `h_a`.** A smaller conv2d stack over the dequantized main latent, producing a hyperprior latent `z`.
4. **TurboQuant forward** on `z`, producing `c_z`.
5. **Hyperprior synthesis `h_s`.** A conv2d stack that reads dequantized `ẑ` and emits two outputs: the per-coordinate codeword logits `π_logits` for the main latent and the log-normal parameters `norm_params` for the per-position norms.
6. **Factorized prior for `c_z`.** A small learned table of marginal codeword probabilities, baked into the `.mll` artifact. This is the entropy model's root — everything else is conditional on it.
7. **Host-side arithmetic coder.** Reads probability tensors from the Manta pipeline and writes the `.mrg` bitstream. Arithmetic coding is serial, byte-granular, and explicitly not a GPU workload; it lives in a Go package during encode and the same source compiled to WASM during decode.
8. **Synthesis net `g_s`.** Symmetric conv2d stack with `conv2d_transpose` for upsampling. Reads `ŷ` and produces `x̂`.

### Encode-side data flow (training)

```
image x  ─►  g_a  ─►  y
                      │
                      ▼
                  TurboQuant  ─►  (c_coords, c_norms)
                      │                   │
                      ▼                   │
                     h_a                  │
                      │                   │
                      ▼                   │
                  TurboQuant  ─►  c_z     │
                                   │      │
               factorized prior ───┤      │
               p(c_z)              │      │
                                   ▼      │
                              h_s ────────┤
                                          ▼
                              (π_logits, norm_params)
                                          │
                                          ▼
                    rate loss: H(c_coords, π_logits)
                             + H(c_norms,  norm_params)
                             + H(c_z,      factorized prior)

   (ŷ, norm) = dequant(c_coords, c_norms) ─►  g_s  ─►  x̂
                                                         │
                                                         ▼
                                            distortion loss: MSE(x, x̂)
```

Total loss is R + λD. Gradients flow through `g_s`, `h_s`, `h_a`, `g_a`, and through both TurboQuant ops via identity straight-through estimators. The quantizer has no learnable parameters; all training updates live in the conv nets and the factorized prior table.

### Encode-side data flow (deployment)

Host Go binary: runs `g_a` + `h_a` + `h_s` on CUDA via Manta, collects `c_z`, `c_coords`, `c_norms`, `π_logits`, and `norm_params`, hands them to a Go-side arithmetic coder, and writes a `.mrg` file. No training code loaded at deploy time — a single `manta run` entry point plus a Go wrapper.

### Decode-side data flow (browser)

```
.mrg file
    │
    ▼
WASM header parser  ────────────►  72-byte header
    │                              (validates magic, version, CRC)
    ▼
WASM arithmetic decoder ──► c_z   (using factorized prior baked into .mll)
    │
    ▼
Manta runtime (WebGPU backend)
    ├── turboquant_decode(c_z) → ẑ
    └── h_s(ẑ) → (π_logits, norm_params)
    │
    ▼
π_logits, norm_params handed back to WASM
    │
    ▼
WASM arithmetic decoder (pass 2, using π_logits) ──► c_coords
WASM arithmetic decoder (pass 3, using norm_params) ──► c_norms
    │
    ▼
Manta runtime (WebGPU backend)
    ├── turboquant_decode(c_coords, c_norms) → ŷ
    └── g_s(ŷ) → x̂
    │
    ▼
x̂ blitted to <canvas>
```

The ping-pong between WASM and WebGPU is intentional: arithmetic decoding is serial and cheap on the CPU side; convolutional work is parallel and belongs on the GPU. Running arithmetic decoding in a compute shader is a standard mistake and is explicitly out of scope.

### Manta pipeline vs host code boundary

**Manta pipelines (`.manta` source, compiled to `.mll`):**
- `analyze(x) -> (c_z, c_coords, c_norms, π_logits, norm_params)` — training and encode-side deployment
- `synthesize_hyperprior(c_z) -> (π_logits, norm_params)` — browser-side between arithmetic decode passes
- `synthesize_image(c_coords, c_norms) -> x_hat` — browser-side after final arithmetic decode

**Host code:**
- `.mrg` file reader and writer, Go on encode side and Go-compiled-to-WASM on decode side, sharing source
- Arithmetic coder, same Go source on both sides
- Image I/O, canvas glue, and the WASM↔WebGPU orchestration in the browser

### Explicitly not in v1

- Perceptual loss heads beyond MS-SSIM
- Any conditional entropy beyond the hyperprior-to-codeword map (no residual blocks, no attention, no masked anything — those are v2 and v3)
- Video, motion compensation, temporal prediction
- Any Manta operators beyond `conv2d`, `conv2d_transpose`, `gdn`, `igdn`, and the TurboQuant quantizer ops, plus what Manta already has

---

## 3. Manta extension surface

Everything below is a first-class Manta deliverable landed as part of v1 work on Mirage, reusable by anything else built on Manta afterward. "All the levers" is the design principle — v1 exposes the full Ballé 2018 configuration surface to training experimentation, not a simplified subset.

### New MIR operators

| Op | Signature sketch | Notes |
|---|---|---|
| `conv2d` | `f16[N, C_in, H, W] × f16[C_out, C_in, kH, kW] -> f16[N, C_out, H', W']` | Configurable kernel, stride, padding, dilation, groups. Forward only at MIR level — backward is a derived op. The centerpiece. |
| `conv2d_transpose` | analogous | For synthesis-net upsampling. |
| `gdn` | `f16[N, C, H, W] -> f16[N, C, H, W]` | First-class fused op, not a composition of pointwise + 1×1 conv. Fused GDN is meaningfully faster and numerically more stable than the composed form, and Ballé 2018 is one of the levers the user wants preserved. |
| `igdn` | analogous | Synthesis-side inverse GDN. |
| `turboquant_encode` | `f16[N, C, H, W] -> (q_bits[N, C, H, W], q_norm[N, H, W])` | Bit-width `const i32` parameter selects 2, 4, or 8. Identity straight-through gradient. |
| `turboquant_decode` | `(q_bits[N, C, H, W], q_norm[N, H, W]) -> f16[N, C, H, W]` | Inverse of encode. Merged with the existing `dequant` intrinsic if the shapes and semantics are compatible; otherwise this is a new op. |
| `cross_entropy_factorized` | takes codeword indices and factorized logits (per-coordinate categorical OR bit-plane factorized), returns scalar rate | Consumed by the R term in the rate-distortion loss. |
| `mse_loss` | standard | First-class for the training path. |
| `ms_ssim_loss` | standard | First-class for MS-SSIM training runs. |

### LIR additions

- Two-dimensional schedule hints: `tile_2d`, `subgroup_2d`, and a halo/border concept for convolution kernel overlap. The existing hint surface is 1D (transformer-shaped); 2D convolution requires expressing blocked tiles with halo exchange.
- A buffer layout concept for NHWC vs NCHW with a per-backend preference. CUDA prefers NCHW; WebGPU prefers NHWC. Manta stays backend-neutral at IR level and picks at lowering time.

### Backward / training path

- A generic autograd path beyond contrastive loss. The existing `embedding_trainer` is tightly coupled to contrastive. v1 extends Manta's training path to support an R + λD loss wrapper, arbitrary pipeline graphs, and gradient flow through `conv2d`, `conv2d_transpose`, `gdn`, `igdn`, the TurboQuant ops (identity STE), `cross_entropy_factorized`, `mse_loss`, `ms_ssim_loss`, and the existing matmul / softmax / normalize / LayerNorm / pointwise ops.
- CUDA backward kernels for all of the above. Each forward op gets a matching backward kernel.
- Checkpointing and optimizer accel reused from the existing embedding-trainer path.

### CUDA backend (`runtime/backends/cuda`)

- Hand-written PTX/CUDA-C forward and backward kernels for `conv2d` and `conv2d_transpose`. Pattern follows the existing matmul kernels. The strided-conv and `groups > 1` paths are where the complexity lives.
- Kernels for `gdn` / `igdn` forward and backward.
- Kernels for `turboquant_encode` / `turboquant_decode` at codec-relevant sizes.

### WebGPU backend (`runtime/backends/webgpu`)

- `conv2d` promoted from host fallback to a device compute shader. **Highest-risk item in v1.** WebGPU device execution is currently "landing" per the Manta status, not shipped, and Mirage v1 needs it to run in the browser.
- `conv2d_transpose` as a device compute shader.
- `gdn` / `igdn` and `turboquant_decode` as device compute shaders.
- The encode path (`turboquant_encode`, backward kernels, `gdn` backward, etc.) does *not* need a WebGPU implementation in v1. WebGPU is decode-only in v1.

### Runtime / host surface

- A host Go package for image I/O: PNG and PPM minimum, JPEG as a stretch target.
- A host Go arithmetic coder that consumes Manta-produced probability tensors and produces `.mrg` bytes. Lives outside Manta proper but next to it as a sibling package.
- The same arithmetic coder compiled to WASM for the browser side via Go's `GOOS=js GOARCH=wasm` target. Encode and decode share source.

### Stretch (defer to v2 if v1 schedule bites)

- Fused GDN + conv kernel (v1 ships separate ops, v2 fuses)
- NHWC / NCHW auto-conversion at pipeline boundaries (v1 requires callers to be layout-aware)
- Additional perceptual losses beyond MS-SSIM

---

## 4. TurboQuant quantizer contract

This section specifies the Manta surface for TurboQuant-as-a-dtype as it behaves inside a trainable image codec. The *interface* described here is frozen across v1/v2/v3; the *implementation* below the interface is open to improvement throughout the project.

### Quantizer characteristics

TurboQuant in the codec has a fixed, structured, data-independent codebook. No learned parameters live in the quantizer itself. Bit-width is configurable at 2, 4, or 8 bits, producing codebooks of 4, 16, or 256 codewords per coordinate respectively. The codec inherits TurboQuant's constant compression ratio and exposes the bit-width as a rate-distortion lever alongside λ.

What the codec learns:

1. The analysis and synthesis nets, which learn to produce latents whose distribution matches TurboQuant's codebook cells.
2. The entropy model, which learns a prior over codeword indices conditional on the hyperprior.

The quantizer itself is structural and untrained. Its behavior at the **implementation** level — rotation kernel, codebook construction, per-coordinate quantization rule, bit-packing layout, CUDA and WebGPU kernels, per-vector norm handling — is explicitly open for improvement throughout v1/v2/v3. Improvements that preserve the interface are additive and silently benefit all three phases. Improvements that change the interface are coordinated changes touching all three phases together.

### Op signatures

```
turboquant_encode(x: f16[N, C, H, W], bits: const i32)
    -> (c_coords: q_bits[N, C, H, W], c_norms: q_norm[N, H, W])

turboquant_decode(c_coords: q_bits[N, C, H, W], c_norms: q_norm[N, H, W])
    -> f16[N, C, H, W]
```

Where `q_bits` resolves to `q2`, `q4`, or `q8` based on the `bits` parameter (Manta already exposes `q4` and `q8`; `q2` lands in v1 if it isn't already there). `q_norm` is a new fixed-width 8-bit log-space scalar dtype introduced in v1 for per-position norms.

### Straight-through estimator

```
∂L/∂x = ∂L/∂turboquant_decode(turboquant_encode(x))
```

Identity STE. The gradient passes through the quantizer as if it were the identity function. No soft-TurboQuant relaxation, no annealed temperature, no stochastic quantization. This is the simplest defensible estimator, matches standard practice in neural codec training, and keeps v1 scope focused. Softer estimators are a research question for future phases.

### Codeword probability interface

The entropy model factorizes over TurboQuant's natural axes. For a rotated latent at bit-width `b` with `C` coordinates at each spatial position, v1 supports two factorization configurations, both sharing the same cross-entropy loss op:

**Per-coordinate categorical (the baseline factorization):**

```
π_logits: f16[N, C, H, W, K]    where K = 2^b
```

Coordinates in the rotated space are modeled as conditionally independent given the hyperprior. This is the natural and standard factorization — a joint distribution over all `K^C` per-position combinations is astronomical.

**Bit-plane factorized (finer factorization):**

```
π_logits: f16[N, C, H, W, b, 2]
```

Each `b`-bit index is decomposed into `b` individual Bernoulli bits, each modeled independently given context. Finer granularity than the per-coordinate categorical form.

Both configurations feed the same host-side arithmetic coder, which is aware of both forms. The loss op `cross_entropy_factorized` consumes whichever form the entropy model emits and produces a scalar rate term. v1 runs rate-distortion measurements with both factorizations and the published headline result is the winner on iso-distortion, or both curves if they are close.

### Norm coder

The per-spatial-position norm tensor (`f32[N, H, W]`, one scalar per latent position) is coded using the same hyperprior `ẑ` that drives the codeword distributions. `h_s` is two-headed:

```
h_s(ẑ) → (π_logits, norm_params)

where:
  π_logits    : f16[N, C, H, W, K]   or   f16[N, C, H, W, b, 2]
  norm_params : f16[N, H, W, 2]
```

`norm_params` parameterizes a **log-normal distribution per spatial position**, with two scalars per position interpreted as `(μ, σ)`. Log-normal is chosen because latent norms are strictly positive, heavy-tailed, and spatially varying across natural images.

The norms themselves are quantized to a fixed, structural 8-bit log-space scalar representation (dtype `q_norm`) with uniform step size in log space. The quantizer step and range are constants baked into the arithmetic coder, not learned parameters.

The norm contribution to the rate loss is `H(c_norms, norm_params)`, summed into the total R term alongside the coordinate and hyperprior rate terms.

### Contract invariants inherited by v2 and v3

1. The quantizer interface is stable across phases. The quantizer implementation is open for improvement.
2. The entropy model is a replaceable submodule whose output conforms to one of the factorized `π_logits` shapes. v2 and v3 swap the entropy model; they do not invent new factorization axes without a coordinated contract change.
3. The loss op is `cross_entropy_factorized`. Never a Gaussian-parameter loss over pre-quantization latents.
4. The arithmetic coder operates on whichever factorized form the current entropy model emits. v1 ships with both per-coordinate categorical and bit-plane factorized; v2 and v3 extend only if the new entropy model requires it.

Invariant (4) is what makes v1 → v2 → v3 additive rather than rewrites. Everything downstream of the entropy model — quantizer, coder, browser runtime — only sees `π_logits` and does not care how they were computed.

### Explicitly not in the v1 contract

- Learned codebooks. TurboQuant's codebook is structural and data-independent; this is a load-bearing design decision, not a simplification.
- Continuous relaxations of the quantizer. No Gumbel-softmax, no soft-vector-quantization, no annealing schedules.
- Per-channel or per-position bit-width selection. v1 picks one global bit-width at training time. Per-position bit-width is a credible v3 extension but out of scope for v1.
- Any interaction between the quantizer and the analysis/synthesis weights during quantizer "training." The quantizer has no training.

---

## 5. Training approach

### Datasets

- **Training.** Vimeo-90k frames extracted as still images plus the CLIC training set. This matches what CompressAI's reference Ballé 2018 trains on; the whole point of v1 as a baseline is to match training distribution with the reference so the only variable left is the quantizer. Random 256×256 patches at load time. A host Go dataloader reads files, decodes, crops, and feeds f16 batches into the Manta training pipeline.
- **Validation.** Kodak (24 images, canonical benchmark). Evaluated every N iterations for early-stopping and checkpoint selection.
- **Test.** Kodak plus CLIC2020 Professional and CLIC2020 Mobile. Final rate-distortion numbers reported on these.

### Rate-distortion loss

```
L(x; λ) = D(x, x̂) + λ · R(c_z, c_coords, c_norms)

where:
  D(x, x̂) = MSE or (1 − MS-SSIM), depending on run
  R       = H(c_z,      factorized_prior_z)
          + H(c_coords, π_logits)
          + H(c_norms,  norm_params)
```

λ is fixed per training run. Each run produces one `(bpp, quality)` point on the final rate-distortion curve.

### λ sweep

Ballé 2018's standard MSE λ grid: `{0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800}`. Eight models per distortion metric, covering ~0.1 bpp to ~2.0 bpp. v1 reproduces this grid faithfully — deviating from the standard grid would make direct comparison to CompressAI's published curves harder. A separate standard grid is used for MS-SSIM runs.

### Distortion metrics

Two training objectives, two separate sets of runs:

- **MSE runs.** Per-pixel mean squared error between `x` and `x̂`. Produces the PSNR rate-distortion curve.
- **MS-SSIM runs.** `(1 − MS-SSIM)` as the distortion loss. Produces the MS-SSIM rate-distortion curve.

### Factorization runs

v1 ships both per-coordinate categorical and bit-plane factorized entropy models. The run ordering:

1. Train both factorizations at three headline λ values (low/mid/high bitrate) on MSE. Six runs.
2. Declare a winner, or confirm they are close enough to warrant the full grid on both.
3. Train the full 8-point λ grid for the winning factorization on MSE (eight runs) and MS-SSIM (eight runs). Sixteen runs.

**Total:** ~22 runs. If the two factorizations are close at the headline λ values, run the full grid for both, yielding 32 runs and the honest answer.

### Optimizer and schedule

Adam with learning rate 1e-4, β₁=0.9, β₂=0.999, no weight decay. Batch size 16 × 256×256 patches. Training length ~2M iterations per run, learning rate decayed 10× for the last 200k iterations. Standard Ballé 2018 recipe. Manta's existing CUDA-accelerated Adam is reused via the generic-autograd extension.

### Training pipeline

```
train_step(x_batch: f16[N, 3, H, W], lambda: f32)
    -> (loss: f32, rate: f32, distortion: f32)
```

Manta's runner loops this pipeline, reads gradients via the autograd path, and applies Adam. Analysis net, hyperprior, and synthesis net weights are the trainable parameters; the quantizer has none.

### Compute budget

Per-run wall clock on a single 4090-class GPU at 2M iterations × batch 16 × 256² patches:

| Regime | Per-run | 22 runs | Notes |
|---|---|---|---|
| Conservative | 1.2–1.6 days | 26–35 days | First-pass kernels, baseline autograd. Roughly iso with mature PyTorch + cuDNN. |
| Expected | 0.5–0.8 days | 11–18 days | Tuned kernels, fused GDN, fused Adam, quantized activations where cheap. Planning target. |
| Aggressive | 0.2–0.35 days | 4–8 days | Fully fused, TurboQuant storage in the forward cache, pinned memory. Ceiling; probably not v1 but possibly end-of-v1. |

On A100/H100-class hardware, scale these down by ~1.5–2×. The Expected regime is what v1 plans against: roughly 1–2 weeks of wall clock for the full 22-run grid on a single 4090, or one week on an A100.

### Gate G1 — conv2d throughput parity

Before kicking off the full λ sweep, v1 measures a single training iteration's wall clock on a representative Ballé 2018 config with the new Manta `conv2d` and `conv2d_transpose` kernels, compared against the same iteration running through CompressAI's reference on the same hardware. The gate passes if Manta is within **1.5× of CompressAI wall clock** and fails otherwise. Failure triggers a kernel-tuning pass before the training grid starts. A slow-conv2d surprise discovered at run 12 of 22 is a nightmare; a two-week kernel-tuning pass before run 1 is cheap by comparison.

### Checkpointing and reproducibility

Checkpoints every 10k iterations plus best-on-validation, stored as Manta's sealed `.mll` package exports so a completed training run produces a ready-to-deploy artifact without conversion. Runs logged to `runs/` with dataset hashes, hyperparameters, λ, factorization choice, and optimizer seeds, matching Manta's existing provenance pattern. Every published rate-distortion point is traceable to its exact training configuration.

---

## 6. Bitstream format (`.mrg` v1)

### File layout

All multi-byte integers are little-endian. All offsets are in bytes.

```
offset  size   field                       notes
------  ----   ----------------------      ----------------------------------------
 0       4     magic                       ASCII "MRG1" (0x4D 0x52 0x47 0x31)
 4       1     format_version              1 for v1
 5       1     flags                       bit 0: distortion metric (0=MSE, 1=MS-SSIM)
                                           bit 1: factorization (0=per-coord categorical,
                                                                 1=bit-plane factorized)
                                           bits 2-7: reserved
 6       1     bit_width                   2, 4, or 8 — TurboQuant bit-width for c_coords
 7       1     reserved_0                  must be 0
 8       4     image_width                 original image width in pixels
12       4     image_height                original image height in pixels
16       4     latent_channels             C dimension of the latent tensor
20       4     latent_height               H dimension of the latent tensor
24       4     latent_width                W dimension of the latent tensor
28      32     model_fingerprint           SHA-256 of the decode-side .mll artifact
60       4     c_z_bytes                   length of the hyperprior codeword stream
64       4     c_coords_bytes              length of the coordinate codeword stream
68       4     c_norms_bytes               length of the norm codeword stream
72       *     c_z_payload                 c_z_bytes of arithmetic-coded data
 *       *     c_coords_payload            c_coords_bytes of arithmetic-coded data
 *       *     c_norms_payload             c_norms_bytes of arithmetic-coded data
END-4    4     crc32                       CRC-32 over all prior bytes
```

Fixed header is 72 bytes. All fields are naturally aligned for direct read from Go on the encoder side and WASM on the decoder side.

### Field semantics

- **`magic`** (`"MRG1"`): identifies the file as a Mirage artifact. Sets the first-byte stream for magic-based content-type detection.
- **`format_version`**: the v1/v2/v3 distinguisher. v1 writes 1. v2 writes 2. v3 writes 3. Decoders select pipelines based on this byte. Across phases the format is additive — v2 may extend the header or add payload streams but does so by incrementing this byte, not reinterpreting existing bits.
- **`flags`**: two bits used in v1 (distortion metric of the training, factorization form of the coordinate stream). Decoders use these to select entropy decoder modes and surface quality metadata to consumers.
- **`bit_width`**: the TurboQuant bit-width the model was trained at. Fixed per file. Future per-position bit-width selection would require a format version bump.
- **`image_width` / `image_height`**: original pixel dimensions.
- **`latent_channels/height/width`**: shape of `ŷ` that the synthesis net expects. Stored explicitly rather than derived.
- **`model_fingerprint`**: SHA-256 of the decode-side `.mll` artifact. The browser checks this against its loaded model before attempting decode. If it doesn't match, the decoder aborts with a clear "model mismatch" error — wrong-model decode produces garbage and there is no graceful fallback.
- **`c_z_bytes / c_coords_bytes / c_norms_bytes`**: stream lengths, stored up front so the decoder can slice the payload without parsing it.
- **`crc32`**: standard CRC-32 over all prior bytes. Strictly a corruption check; cryptographic integrity is the transport layer's job.

### Arithmetic coder streams

All three payloads are produced by the same Go range coder (compiled to WASM for the decoder). Standard adaptive range coder with 32-bit state. What matters is that encoder and decoder run bit-identical logic, which is trivially ensured by sharing Go source across both targets.

- **`c_z_payload`** uses the factorized prior `p(c_z)`, a small learned table baked into the `.mll` artifact. No conditioning.
- **`c_coords_payload`** uses `π_logits` produced by running `synthesize_hyperprior(ẑ)` after `c_z` is decoded. Conditioned on the already-decoded hyperprior.
- **`c_norms_payload`** uses `norm_params` produced by the same `synthesize_hyperprior` call (`h_s` is two-headed). Conditioned on the already-decoded hyperprior.

The WASM ↔ Manta/WebGPU ping-pong happens exactly once per file: one GPU dispatch for the hyperprior synthesis, one for the main image synthesis.

### Evolution rules across phases

- **v1 → v2.** Bump `format_version` to 2. v2 may add payload streams, extend the header, or add flag bits. v2 decoders must handle v1 files. v1 decoders reject v2 files cleanly.
- **v1 → v3.** Same rule.
- **Never**: reinterpret existing bits, reuse flag positions, change existing field offsets, change endianness. The header layout is a hard constraint for the lifetime of `format_version = 1`.

### Endianness and alignment

Little-endian throughout. All fields at natural alignment for `Uint32Array` / `Uint8Array` reads from JavaScript or Rust/Zig WASM. The Go encoder and the WASM decoder share a `format.go` file that defines the struct layout once.

### Metadata extension point

A structured metadata block (color profile, creation time, source attribution) is a common codec feature. v1 explicitly does not ship one — the 72-byte header is the entire header. v2 may add an optional metadata section between the fixed header and the payloads, gated by a reserved flag bit.

---

## 7. Browser decoder

### Stack

```
┌──────────────────────────────────────────────────────────────┐
│ Demo HTML page (index.html + minimal CSS + one canvas)       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ TypeScript wrapper (@mirage/decoder)                         │
│   - fetches .mrg files                                        │
│   - fetches .mll artifacts once, caches in IndexedDB          │
│   - calls into the WASM runtime                               │
│   - receives RGB output, blits to <canvas>                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Manta runtime compiled to Go WASM                            │
│   - loads .mll artifact                                       │
│   - parses .mrg header, runs arithmetic decoder               │
│   - dispatches Manta pipelines to WebGPU                      │
│   - manages GPU buffer lifecycle via WebGPU JS interop        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ WebGPU compute shaders (WGSL, emitted from Manta LIR)        │
│   - conv2d, conv2d_transpose                                  │
│   - gdn, igdn                                                 │
│   - turboquant_decode (4-bit and 8-bit)                       │
│   - pointwise ops (LayerNorm, activations)                    │
└──────────────────────────────────────────────────────────────┘
```

Arithmetic decoding and header parsing live in the Go/WASM layer because Manta's runtime is already there — writing them separately in hand-rolled JavaScript would duplicate code that already needs to exist in Go for the encoder. The entire host side of the pipeline is reused on the decode side by compiling the same Go source to WASM.

### Go → WASM toolchain

Manta is pure Go, so `GOOS=js GOARCH=wasm go build ./runtime/browser` produces a `.wasm` binary containing the runtime, the WebGPU backend, the arithmetic coder, and the `.mrg` parser. Expected binary size: ~5–10 MB after TinyGo build with tree-shaking, ~15–20 MB with vanilla Go's WASM build. Loaded once per session, cached in the browser's HTTP cache, reused for every decode.

### WebGPU compute shaders (v1 deliverables)

1. **`conv2d` forward (WGSL).** Tile-based with workgroup-local memory for weight reuse. Specialized shader variants for 3×3 and 5×5 kernels (Ballé 2018 uses only small kernels) plus a general-kernel variant. Strided and dilated variants.
2. **`conv2d_transpose` forward (WGSL).** Same tile pattern, for synthesis-net upsampling.
3. **`igdn` forward (WGSL).** Per-channel divisive normalization with channel-interaction matrix. Workgroup-local accumulation for the divisor term.
4. **`turboquant_decode` forward (WGSL)** for 4-bit and 8-bit paths. Takes coordinate indices and per-position norms, runs inverse rotation, produces dequantized latent in f16. The inverse Walsh-Hadamard rotation is the expensive part and gets its own compute shader.
5. **Pointwise ops.** Activations, LayerNorm, elementwise arithmetic, emitted from the shared pointwise lowering path.

Forward-only is the v1 browser deliverable; training WebGPU kernels are out of scope. Training runs on CUDA.

### `.mll` artifact loading

First visit: TypeScript wrapper fetches the decoder `.mll` (expected ~25–60 MB depending on λ and channel count), validates its SHA-256, stores it in IndexedDB keyed by fingerprint, hands it to the Go/WASM runtime.

Subsequent visits: wrapper reads from IndexedDB, validates, skips fetch. Multiple `.mll` artifacts (one per λ × factorization × distortion combo) may coexist in the cache under an LRU policy.

### Decode flow

```
1.  User drops a .mrg file on the canvas
2.  TS wrapper fetches .mrg bytes
3.  TS wrapper reads the 72-byte header, extracts model_fingerprint
4.  TS wrapper checks IndexedDB for that fingerprint's .mll
    (fetches if absent, validates, caches)
5.  TS wrapper hands .mrg bytes + loaded .mll handle to Go/WASM runtime
6.  Go/WASM runtime validates CRC-32
7.  Go/WASM runtime runs arithmetic decode pass 1: c_z using factorized
    prior baked into the .mll
8.  Go/WASM runtime dispatches synthesize_hyperprior(c_z) to WebGPU
    - uploads c_z to GPU
    - Manta pipeline runs (conv2d stack + igdn + pointwise)
    - reads back (π_logits, norm_params)
9.  Go/WASM runtime runs arithmetic decode pass 2: c_coords using π_logits
10. Go/WASM runtime runs arithmetic decode pass 3: c_norms using norm_params
11. Go/WASM runtime dispatches synthesize_image(c_coords, c_norms) to WebGPU
    - turboquant_decode produces dequantized latent
    - Manta pipeline runs (conv2d_transpose stack + igdn + pointwise)
    - reads back x_hat
12. Go/WASM runtime returns x_hat to TS wrapper
13. TS wrapper converts x_hat to ImageData (clipping to [0,255] u8)
14. TS wrapper blits ImageData to <canvas>
```

Steps 2–3 and 5–12 run asynchronously against the UI thread. Two GPU ping-pongs total per decode.

### Performance target

For a 768×512 Kodak-class image at 4-bit, `C=192` channels, Ballé 2018 topology, on a mid-range 2024 laptop GPU (Intel Arc, Apple M2, RTX 3050-class):

- **Target:** ≤ 250 ms end-to-end from "user drops file" to "pixels on canvas"
- **Breakdown:** ~30 ms fetch+parse+CRC, ~5 ms arithmetic decode pass 1, ~40 ms WebGPU hyperprior synthesis, ~10 ms arithmetic decode passes 2 and 3, ~120 ms WebGPU synthesis pass, ~20 ms readback + canvas blit
- **Stretch:** ≤ 150 ms, achievable after v1 kernel-tuning or early v2

### Failure modes

1. **No WebGPU support.** `navigator.gpu` feature check at page load. If absent, the page displays: "Mirage decode requires WebGPU (Chrome 113+, Edge 113+, Safari 17.4+, Firefox Nightly). No silent fallback." The CPU WASM fallback for conv2d would be ~100× slower and would undermine the "good to great" Manta story.
2. **Corrupted `.mrg`.** CRC-32 mismatch aborts decode with "file corrupted, not a valid Mirage file."
3. **Model fingerprint mismatch.** Wrapper attempts to fetch the matching `.mll` from a static-hosted fingerprint → URL registry. If unknown, aborts with "this file was encoded with a model this viewer doesn't have."
4. **WebGPU device loss mid-decode.** Manta runtime catches the error, releases GPU buffers, retries once with a fresh adapter. Second failure aborts with "GPU became unavailable; reload to try again."
5. **Memory pressure on very large images.** Demo page checks input dimensions and refuses anything above a configurable cap (default 2048×2048) with a clear message rather than silently OOMing the tab.

### Demo page

A single-file HTML page: drop zone, canvas, a small "about" paragraph, a link to the m31labs writeup, and three hosted demo files (low/mid/high bitrate). No framework, no build step — one HTML + one TS compiled to one JS + two WASM + the `.mll` artifacts. Total page weight: one 5–10 MB runtime binary, one ~30 MB model artifact (cached after first load), a few KB of HTML/JS. The page demonstrates the decoder; richer UI is out of v1 scope.

---

## 8. Evaluation and testing methodology

### Benchmark corpora

- **Kodak** (24 images, canonical). Primary rate-distortion corpus. Most published neural image codec numbers exist on Kodak, which makes direct comparison trivial.
- **CLIC2020 Professional Test Set** (~250 high-resolution images, professional photography). Secondary corpus, better test of high-bitrate regime and diverse content.
- **CLIC2020 Mobile Test Set** (~250 mobile-phone images). Third corpus, tests noisier / lower-quality-source regime.

### Quality metrics

- **PSNR** (peak signal-to-noise ratio, dB). Reported per-image and averaged. Paired with MSE-trained models.
- **MS-SSIM** (multi-scale structural similarity, 0–1 or dB). Reported per-image and averaged. Paired with MS-SSIM-trained models. Both linear and `-10·log₁₀(1 − MS-SSIM)` forms reported — the log form is standard in codec papers.
- **bpp** (bits per pixel). File size divided by pixel count. The rate axis of every rate-distortion plot.

### BD-rate (Bjøntegaard delta rate)

Standard in codec literature. Measures the average bitrate savings of one codec over another at iso-quality across a rate-distortion curve. Computed between Mirage v1 and each baseline. A BD-rate of -10% means Mirage v1 uses 10% fewer bits than the baseline to achieve the same quality; positive values are worse. BD-rate is reported against both PSNR and MS-SSIM.

### Baseline systems

| Baseline | What it tests | Comparison framing |
|---|---|---|
| **CompressAI reference Ballé 2018** with standard scalar quantizer | The core claim: TurboQuant vs additive-uniform-noise scalar at iso-architecture | The headline comparison. Same conv net topology, same training corpus, same λ grid. Only the quantizer changes. Any gap is attributable to the quantizer. |
| **JPEG** | Classical codec floor | Sanity baseline; expected to lose badly. |
| **JPEG 2000** (OpenJPEG) | Wavelet-based classical codec | Still a relevant reference. |
| **WebP** (libwebp) | Web-canonical lossy codec | Meaningful comparison for web context. |
| **BPG** (libbpg, H.265 intra) | Current ceiling for non-neural still-image codecs | Strong baseline. v2 targets this; v1 is not expected to match it. |
| **VTM intra** (H.266 reference, optional) | State-of-the-art non-neural codec | Stretch comparison. v3 target. |

### What "done" quantitatively means for v1

- Matches or beats CompressAI reference Ballé 2018 rate-distortion on Kodak at both PSNR and MS-SSIM, across the full λ sweep. **This is the publishable claim.** Equivalent BD-rate counts as success; a modest improvement is the aspiration.
- Produces a complete rate-distortion curve on Kodak, CLIC2020 Professional, and CLIC2020 Mobile.
- Produces a working WebGPU browser decoder that decodes any file in the test corpora within the 250 ms target on representative mid-range hardware.
- Documents the Manta extension surface (every new op, every kernel, every autograd extension) as reusable first-class Manta features.

### Testing strategy

**Unit tests (per Manta op, per backend).**

- Known-answer forward tests: each op (`conv2d`, `conv2d_transpose`, `gdn`, `igdn`, `turboquant_encode`, `turboquant_decode`, `cross_entropy_factorized`, `mse_loss`, `ms_ssim_loss`) has a known-answer test against a reference computed with a naive pure-Go implementation, for both the CUDA and WebGPU backends.
- Backward correctness tests: gradient checks against numerical finite-differencing for every backward kernel, at small shapes.
- Layout independence tests: the same forward op produces numerically-matching output on NHWC and NCHW layouts.

**Numerical parity tests (Mirage vs CompressAI).**

- For a fixed image and a fixed λ, Mirage v1 and CompressAI reference Ballé 2018 produce reconstructions whose PSNR differs by less than 0.5 dB (accounting for natural variance from different training seeds and the quantizer change). Large divergences indicate a bug.

**Regression tests (training convergence).**

- A small synthetic corpus (~100 random images) trains to a known loss within tolerance in a fixed number of iterations. Used as CI regression for the autograd path and kernel correctness.

**End-to-end tests (codec correctness).**

- For a curated small test set, `encode(x) → .mrg → decode(.mrg) → x̂` completes without error and produces the expected PSNR/MS-SSIM numbers.
- Encode and decode run bit-identical arithmetic coder state (verified by round-tripping the codeword indices through just the coder without the network).

**Kernel performance tests (Gate G1 and regressions).**

- Gate G1 measures `conv2d` + `conv2d_transpose` wall clock against CompressAI reference on the same hardware and fails the build if Manta is more than 1.5× slower.
- Continuous perf regression: kernel throughput is measured on a fixed representative shape at every merge and any regression >10% blocks the merge.

**Browser decode correctness.**

- The Go/WASM decoder and the host Go encoder produce identical reconstructions for a fixed test corpus. Any drift between host Go and WASM indicates a compilation-target divergence (e.g., floating-point determinism issues), which is tracked as a blocker.

### Publication format

Each phase gets one m31labs writeup. v1's writeup includes:

- The three rate-distortion curves (Kodak, CLIC Pro, CLIC Mobile) vs all baselines
- The headline BD-rate table: Mirage v1 vs CompressAI reference Ballé 2018 at iso-architecture
- The factorization comparison: per-coordinate categorical vs bit-plane factorized
- The browser decoder demo live at a URL
- The Manta extension surface as an appendix: what landed in Manta, reusable for future projects

---

## Appendix A — Strategic context

v1 is the first phase of a longer arc. The image codec is the forcing function for landing `conv2d`, TurboQuant-native entropy models, generic autograd, and WebGPU device kernels in Manta. Those Manta primitives unlock a sequence of follow-on projects that are the real reason Mirage v1 exists in the shape it does.

### What falls out once v1 ships

- **A GoSX-native image format, immediately.** `<Mirage src="foo.mrg" />` as a first-class GoSX component with a Go server-side encoder and a Go-WASM client-side decoder. No external library, no browser codec dependency, no JPEG/WebP/AVIF patent question, no WebCodecs API gymnastics. The component and its runtime ship inside the GoSX binary itself because the entire thing is Go.
- **A unified runtime foundation.** The Go-WASM runtime binary built in v1 is reused unchanged for every future Mirage codec in the family. One runtime, one loader, one browser glue, many codecs.
- **Patent-free, license-free, Go-native all the way down.** Zero patent exposure, zero vendor dependency, full freedom to ship commercially inside GoSX apps.

### What is a focused additional project

**Mirage Video (`.mrv`).** Structurally Mirage Image v3 plus one temporal prediction network that predicts the current frame's latent from the previous frame's latent, plus a residual entropy model that codes the prediction residual using the same TurboQuant + factorized `π_logits` machinery the image codec already uses, plus a GOP structure in the bitstream (I-frames are image-codec frames; P-frames are residual frames), plus temporal loss terms in the training path (reconstruction + rate over sequences of frames). No new Manta primitives beyond what Mirage Image v1 already landed. Best modeled as "phase 4": a discrete follow-on project to this spec, not a moonshot.

**Mirage Stream (`.mrs`).** Container and manifest format for adaptive streaming on top of Mirage Video. Chunked bitstream framing for incremental decode, I-frame boundaries for seek, a bit-rate ladder for adaptive streaming, a manifest format like HLS/DASH but Mirage-native, server-side segment delivery. Container-format design on top of the codec, not codec research. Well-trodden ground (fMP4 is the reference), doable in Go. Its own spec when the time comes.

**CorkscrewDB-backed semantic video search.** Because Mirage produces quantized latent vectors as a side effect of encoding and CorkscrewDB stores quantized vectors natively with nearest-neighbor queries, every frame of every video can be indexed at encode time with essentially zero additional cost. Point queries like "find the frame in this video that looks most like this reference image" become trivial. Neither H.265 nor AV1 can do this — their latents are neither semantically meaningful nor retained for retrieval. This feature only makes sense in a stack where the codec, the vector database, and the ML runtime are the same project, which describes exactly this stack.

**Edge-side encode + browser-side decode, all in Go.** Because Manta's backend set includes CUDA, Metal, Vulkan, DirectML, and WebGPU from the same `.manta` source, a Mirage Video encoder can run on a phone, Raspberry Pi, or edge node, and decode in any browser, with the same binary family on both sides. Live-streaming use cases (video chat, first-party broadcasts, self-hosted streams) have an entirely different shape when there is no H.264 hardware encoder dependency and no WebRTC required. GoSX could ship a `<MirageLive>` component that connects to a camera, encodes in-process with Manta+CUDA, and streams to a browser decoder over plain WebSocket or QUIC.

### What is aspirational

- **Matching AV1 rate-distortion on video.** Mirage Image v3 should be competitive with H.265 on images. Video is a harder game; AV1 has spent years tuning its P-frame machinery. Matching AV1 BD-rate on video is a multi-year research program — not impossible for a Manta-native codec that exploits TurboQuant aggressively, but not a free addition to Mirage Video either. The honest claim is "competitive for web use cases where licensing matters more than the last 10% of rate-distortion, and genuinely better where unified image/video tooling matters."
- **Real-time encoding throughput.** Software neural video encoders currently run below real-time at HQ settings. Getting to 30 fps live encode on commodity hardware is a kernel-tuning program of its own, and its feasibility depends entirely on how aggressively Manta's kernels can be tuned for this specific workload.
- **Browser vendor standardization.** Proposing Mirage Video as a WICG or W3C draft once Mirage Image v3 is a published research result, the reference implementation is Go-native and runs in a WebGPU decoder, and benchmarks are credible. A long-shot outcome but not absurd.

### The throughline

Mirage Image v1 is chosen over every easier alternative because the Manta primitives it forces into existence are exactly the primitives every other project in this appendix needs. A simpler v1 that avoids conv2d or runs on existing ML tooling would not unlock any of the follow-on work. The whole point of building the hard version is that the hard version is the foundation.
