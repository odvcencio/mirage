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

Remaining work to reach the publishable Balle 2018 result:

- CUDA forward/backward kernels promoted from host-reference execution
- WebGPU device kernels promoted from host-reference execution
- full generic autograd for rate-distortion training
- production `.mll` artifacts that replace the Go patch model fingerprint
- hyperprior synthesis and production learned probabilities instead of the
  current default uniform executable model

The implementation boundary is intentional: the `.mrg` file substrate and host
codec APIs live here, while the learned analysis/synthesis network belongs in
Manta.

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
