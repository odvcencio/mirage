package mirage

import (
	"bytes"
	"testing"
)

func TestArithmeticCoderRoundTripUniform(t *testing.T) {
	model, err := UniformCDF(5)
	if err != nil {
		t.Fatalf("UniformCDF: %v", err)
	}
	symbols := []uint16{0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 2, 2, 4, 1}
	encoded, err := EncodeSymbols(symbols, model)
	if err != nil {
		t.Fatalf("EncodeSymbols: %v", err)
	}
	decoded, err := DecodeSymbols(encoded, len(symbols), model)
	if err != nil {
		t.Fatalf("DecodeSymbols: %v", err)
	}
	if len(decoded) != len(symbols) {
		t.Fatalf("decoded len = %d want %d", len(decoded), len(symbols))
	}
	for i := range decoded {
		if decoded[i] != symbols[i] {
			t.Fatalf("decoded[%d] = %d want %d", i, decoded[i], symbols[i])
		}
	}
}

func TestArithmeticCoderRoundTripSkewed(t *testing.T) {
	model, err := NewCDF([]uint32{30, 1, 3, 9, 2})
	if err != nil {
		t.Fatalf("NewCDF: %v", err)
	}
	symbols := []uint16{0, 0, 0, 3, 0, 2, 0, 4, 1, 0, 3, 3, 0, 2, 0, 0}
	encoded, err := EncodeSymbols(symbols, model)
	if err != nil {
		t.Fatalf("EncodeSymbols: %v", err)
	}
	decoded, err := DecodeSymbols(encoded, len(symbols), model)
	if err != nil {
		t.Fatalf("DecodeSymbols: %v", err)
	}
	for i := range decoded {
		if decoded[i] != symbols[i] {
			t.Fatalf("decoded[%d] = %d want %d", i, decoded[i], symbols[i])
		}
	}
}

func TestArithmeticCoderBytesRoundTrip(t *testing.T) {
	input := []byte("mirage range coder smoke payload")
	encoded, err := EncodeBytes(input)
	if err != nil {
		t.Fatalf("EncodeBytes: %v", err)
	}
	decoded, err := DecodeBytes(encoded, len(input))
	if err != nil {
		t.Fatalf("DecodeBytes: %v", err)
	}
	if !bytes.Equal(decoded, input) {
		t.Fatalf("DecodeBytes = %q want %q", decoded, input)
	}
}

func TestArithmeticCoderRejectsZeroProbabilitySymbol(t *testing.T) {
	model, err := NewCDF([]uint32{1, 0, 1})
	if err != nil {
		t.Fatalf("NewCDF: %v", err)
	}
	if _, err := EncodeSymbols([]uint16{1}, model); err == nil {
		t.Fatalf("EncodeSymbols accepted zero-probability symbol")
	}
}
