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

func TestArithmeticCoderRoundTripPerSymbolModels(t *testing.T) {
	modelA, err := NewCDF([]uint32{20, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	modelB, err := NewCDF([]uint32{1, 20, 1})
	if err != nil {
		t.Fatal(err)
	}
	modelC, err := NewCDF([]uint32{1, 1, 20})
	if err != nil {
		t.Fatal(err)
	}
	symbols := []uint16{0, 1, 2, 0, 2, 1}
	models := []CDF{modelA, modelB, modelC, modelA, modelC, modelB}
	encoded, err := EncodeSymbolsWithModels(symbols, models)
	if err != nil {
		t.Fatalf("EncodeSymbolsWithModels: %v", err)
	}
	decoded, err := DecodeSymbolsWithModels(encoded, models)
	if err != nil {
		t.Fatalf("DecodeSymbolsWithModels: %v", err)
	}
	for i := range decoded {
		if decoded[i] != symbols[i] {
			t.Fatalf("decoded[%d] = %d want %d", i, decoded[i], symbols[i])
		}
	}
}

func TestCDFFromLogitsPreservesLikelySymbols(t *testing.T) {
	model, err := CDFFromLogits([]float32{6, 0, -1}, 64)
	if err != nil {
		t.Fatalf("CDFFromLogits: %v", err)
	}
	if model.Counts[0] <= model.Counts[1] || model.Total != 64 {
		t.Fatalf("unexpected CDF from logits: %+v", model)
	}
}

func TestCDFsFromLogitsBuildsSequence(t *testing.T) {
	models, err := CDFsFromLogits([]float32{
		5, 0, -1,
		-1, 0, 5,
	}, 3, 64)
	if err != nil {
		t.Fatalf("CDFsFromLogits: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models len = %d want 2", len(models))
	}
	if models[0].Counts[0] <= models[0].Counts[1] {
		t.Fatalf("first model is not peaked at symbol 0: %+v", models[0])
	}
	if models[1].Counts[2] <= models[1].Counts[1] {
		t.Fatalf("second model is not peaked at symbol 2: %+v", models[1])
	}
}

func TestNormCDFsFromLogNormalParams(t *testing.T) {
	models, err := NormCDFsFromLogNormalParams([]float32{0, 0.5, 2, 0.75}, 1024)
	if err != nil {
		t.Fatalf("NormCDFsFromLogNormalParams: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models len = %d want 2", len(models))
	}
	if models[0].SymbolCount() != 256 || models[0].Total != 1024 {
		t.Fatalf("unexpected norm model: symbols=%d total=%d", models[0].SymbolCount(), models[0].Total)
	}
	zeroNormSymbol := QuantizeNorm(1)
	if models[0].Counts[zeroNormSymbol] == 0 {
		t.Fatalf("norm model has no mass near exp(mu)")
	}
}
