package mirage

import (
	"math"
	"testing"
)

func TestNormQuantizerMonotonic(t *testing.T) {
	values := []float32{1e-6, 1e-3, 1, 10, 1000}
	var prev byte
	for i, v := range values {
		q := QuantizeNorm(v)
		if i != 0 && q <= prev {
			t.Fatalf("QuantizeNorm(%g) = %d, previous %d", v, q, prev)
		}
		prev = q
	}
}

func TestNormQuantizerRoundTripWithinLogStep(t *testing.T) {
	values := []float32{1e-5, 0.01, 1, 37.5, 10000}
	step := (DefaultNorms.LogMax - DefaultNorms.LogMin) / 255
	for _, v := range values {
		q := QuantizeNorm(v)
		got := DequantizeNorm(q)
		logErr := math.Abs(math.Log(float64(got)) - math.Log(float64(v)))
		if logErr > step/2+1e-6 {
			t.Fatalf("norm %g round-tripped to %g with log error %g > step/2 %g", v, got, logErr, step/2)
		}
	}
}

func TestNormQuantizerClamps(t *testing.T) {
	if got := QuantizeNorm(0); got != 0 {
		t.Fatalf("QuantizeNorm(0) = %d want 0", got)
	}
	if got := QuantizeNorm(float32(math.Inf(1))); got != 255 {
		t.Fatalf("QuantizeNorm(+Inf) = %d want 255", got)
	}
}
