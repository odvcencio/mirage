package mirage

import (
	"bytes"
	"testing"
)

func TestLatentPayloadRoundTripCategorical(t *testing.T) {
	code := testLatentCode(t, 4)
	payloads, err := EncodeLatentPayloads(code, FactorizationCategorical)
	if err != nil {
		t.Fatalf("EncodeLatentPayloads: %v", err)
	}
	decoded, err := DecodeLatentPayloads(payloads, code.Shape, code.BitWidth, FactorizationCategorical)
	if err != nil {
		t.Fatalf("DecodeLatentPayloads: %v", err)
	}
	if !bytes.Equal(decoded.Coordinates, code.Coordinates) || !bytes.Equal(decoded.Norms, code.Norms) {
		t.Fatal("categorical payload did not round-trip")
	}
}

func TestLatentPayloadRoundTripBitPlane(t *testing.T) {
	code := testLatentCode(t, 2)
	payloads, err := EncodeLatentPayloads(code, FactorizationBitPlane)
	if err != nil {
		t.Fatalf("EncodeLatentPayloads: %v", err)
	}
	decoded, err := DecodeLatentPayloads(payloads, code.Shape, code.BitWidth, FactorizationBitPlane)
	if err != nil {
		t.Fatalf("DecodeLatentPayloads: %v", err)
	}
	if !bytes.Equal(decoded.Coordinates, code.Coordinates) || !bytes.Equal(decoded.Norms, code.Norms) {
		t.Fatal("bit-plane payload did not round-trip")
	}
}

func TestLatentPayloadRoundTripWithPerSymbolModels(t *testing.T) {
	code := testLatentCode(t, 2)
	coords, err := UnpackCoordinateSymbols(code)
	if err != nil {
		t.Fatalf("UnpackCoordinateSymbols: %v", err)
	}
	coordinateModels := make([]CDF, len(coords))
	for i, sym := range coords {
		coordinateModels[i] = mustBiasedCDF(t, 1<<code.BitWidth, int(sym))
	}
	normModels := make([]CDF, len(code.Norms))
	for i, norm := range code.Norms {
		normModels[i] = mustBiasedCDF(t, 256, int(norm))
	}
	models := LatentPayloadModels{
		CoordinateModels: coordinateModels,
		NormModels:       normModels,
	}
	payloads, err := EncodeLatentPayloadsWithModels(code, FactorizationCategorical, models)
	if err != nil {
		t.Fatalf("EncodeLatentPayloadsWithModels: %v", err)
	}
	decoded, err := DecodeLatentPayloadsWithModels(payloads, code.Shape, code.BitWidth, FactorizationCategorical, models)
	if err != nil {
		t.Fatalf("DecodeLatentPayloadsWithModels: %v", err)
	}
	if !bytes.Equal(decoded.Coordinates, code.Coordinates) || !bytes.Equal(decoded.Norms, code.Norms) {
		t.Fatal("per-symbol model payload did not round-trip")
	}
}

func TestCoordinateSymbolPackingRoundTrip(t *testing.T) {
	shape := LatentShape{Channels: 5, Height: 2, Width: 1}
	symbols := []uint16{0, 1, 2, 3, 1, 3, 2, 1, 0, 2}
	packed, err := PackCoordinateSymbols(symbols, shape, 2)
	if err != nil {
		t.Fatalf("PackCoordinateSymbols: %v", err)
	}
	code := LatentCode{
		Shape:       shape,
		BitWidth:    2,
		Coordinates: packed,
		Norms:       make([]byte, shape.Positions()),
	}
	got, err := UnpackCoordinateSymbols(code)
	if err != nil {
		t.Fatalf("UnpackCoordinateSymbols: %v", err)
	}
	for i := range got {
		if got[i] != symbols[i] {
			t.Fatalf("symbol[%d] = %d want %d", i, got[i], symbols[i])
		}
	}
}

func TestEncodeDecodeMRGBitPlane(t *testing.T) {
	img, err := NewRGBImage(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := range img.Pix {
		img.Pix[i] = float32(i%17) / 16
	}
	file, err := Encode(img, EncodeOptions{BitWidth: 2, Factorization: FactorizationBitPlane})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if file.Header.Factorization() != FactorizationBitPlane {
		t.Fatal("expected bit-plane header")
	}
	if _, err := Decode(file, DefaultDecodeOptions()); err != nil {
		t.Fatalf("Decode: %v", err)
	}
}

func testLatentCode(t *testing.T, bitWidth int) LatentCode {
	t.Helper()
	shape := LatentShape{Channels: 8, Height: 2, Width: 2}
	latents := make([]float32, shape.Elements())
	for i := range latents {
		latents[i] = float32(i%11) / 5
	}
	code, err := EncodeLatents(latents, shape, bitWidth, 7)
	if err != nil {
		t.Fatalf("EncodeLatents: %v", err)
	}
	return code
}

func mustBiasedCDF(t *testing.T, alphabet int, hot int) CDF {
	t.Helper()
	counts := make([]uint32, alphabet)
	for i := range counts {
		counts[i] = 1
	}
	counts[hot] += 31
	model, err := NewCDF(counts)
	if err != nil {
		t.Fatalf("NewCDF: %v", err)
	}
	return model
}
