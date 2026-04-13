// Package mirage contains the host-side substrate for Mirage Image v1.
//
// Mirage's trainable Manta networks are specified outside this repository, but
// the bitstream, entropy-coder plumbing, image I/O, and TurboQuant latent
// representation can live next to TurboQuant today. The code here follows the
// v1 .mrg contract from docs/superpowers/specs/2026-04-12-mirage-v1-design.md.
package mirage
