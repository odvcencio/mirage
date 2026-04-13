package mirage

import (
	"fmt"
	"math"
)

const (
	defaultNormLogMin = -16.0
	defaultNormLogMax = 16.0
)

// NormQuantizer maps positive latent norms to the q_norm byte described by the
// v1 spec: uniform buckets in log space, with fixed non-learned range.
type NormQuantizer struct {
	LogMin float64
	LogMax float64
}

// DefaultNorms is the structural v1 q_norm mapping used by helpers in this
// package. It covers roughly [1e-7, 8.9e6].
var DefaultNorms = NormQuantizer{
	LogMin: defaultNormLogMin,
	LogMax: defaultNormLogMax,
}

func (q NormQuantizer) validate() error {
	if math.IsNaN(q.LogMin) || math.IsNaN(q.LogMax) || math.IsInf(q.LogMin, 0) || math.IsInf(q.LogMax, 0) {
		return fmt.Errorf("mirage: invalid norm quantizer range")
	}
	if q.LogMax <= q.LogMin {
		return fmt.Errorf("mirage: norm quantizer LogMax must exceed LogMin")
	}
	return nil
}

// Quantize maps a positive norm to one q_norm byte. Non-positive norms clamp
// to the lowest bucket.
func (q NormQuantizer) Quantize(norm float32) byte {
	if err := q.validate(); err != nil {
		panic(err)
	}
	if norm <= 0 || math.IsNaN(float64(norm)) {
		return 0
	}
	if math.IsInf(float64(norm), 1) {
		return 255
	}
	logNorm := math.Log(float64(norm))
	t := (logNorm - q.LogMin) / (q.LogMax - q.LogMin)
	if t <= 0 {
		return 0
	}
	if t >= 1 {
		return 255
	}
	return byte(math.Round(t * 255))
}

// Dequantize maps a q_norm byte back to the center of its log-space bucket.
func (q NormQuantizer) Dequantize(encoded byte) float32 {
	if err := q.validate(); err != nil {
		panic(err)
	}
	t := float64(encoded) / 255
	return float32(math.Exp(q.LogMin + t*(q.LogMax-q.LogMin)))
}

// QuantizeNorm maps a positive norm to one q_norm byte with DefaultNorms.
func QuantizeNorm(norm float32) byte {
	return DefaultNorms.Quantize(norm)
}

// DequantizeNorm maps a q_norm byte back to an approximate positive norm.
func DequantizeNorm(encoded byte) float32 {
	return DefaultNorms.Dequantize(encoded)
}

// QuantizeNorms maps many positive norms to q_norm bytes.
func QuantizeNorms(norms []float32) []byte {
	out := make([]byte, len(norms))
	for i, norm := range norms {
		out[i] = QuantizeNorm(norm)
	}
	return out
}

// DequantizeNorms maps q_norm bytes back to approximate positive norms.
func DequantizeNorms(encoded []byte) []float32 {
	out := make([]float32, len(encoded))
	for i, q := range encoded {
		out[i] = DequantizeNorm(q)
	}
	return out
}
