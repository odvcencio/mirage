package mirage

import (
	"fmt"

	turboquant "github.com/odvcencio/turboquant"
)

// LatentPayloadModels contains the factorized entropy models used to turn a
// TurboQuant latent code into the .mrg payload streams.
type LatentPayloadModels struct {
	Coordinate          CDF
	CoordinateModels    []CDF
	CoordinateBit       CDF
	CoordinateBitModels []CDF
	Norm                CDF
	NormModels          []CDF
}

// DefaultLatentPayloadModels returns uniform v1 models. Learned Manta paths set
// the per-symbol model slices from probability tensors converted by CDFsFrom*.
func DefaultLatentPayloadModels(bitWidth int) (LatentPayloadModels, error) {
	if err := validateMirageBitWidth(bitWidth); err != nil {
		return LatentPayloadModels{}, err
	}
	coordinate, err := UniformCDF(1 << bitWidth)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	coordinateBit, err := UniformCDF(2)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	norm, err := UniformCDF(256)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	return LatentPayloadModels{Coordinate: coordinate, CoordinateBit: coordinateBit, Norm: norm}, nil
}

// EncodeLatentPayloads entropy-codes TurboQuant coordinates and q_norm values
// into the .mrg payload streams using default factorized models.
func EncodeLatentPayloads(code LatentCode, factorization Factorization) (Payloads, error) {
	models, err := DefaultLatentPayloadModels(code.BitWidth)
	if err != nil {
		return Payloads{}, err
	}
	return EncodeLatentPayloadsWithModels(code, factorization, models)
}

// EncodeLatentPayloadsWithModels is the learned entropy-model boundary. The
// caller supplies CDFs; Mirage supplies deterministic arithmetic coding.
func EncodeLatentPayloadsWithModels(code LatentCode, factorization Factorization, models LatentPayloadModels) (Payloads, error) {
	if err := code.Validate(); err != nil {
		return Payloads{}, err
	}
	coords, err := UnpackCoordinateSymbols(code)
	if err != nil {
		return Payloads{}, err
	}
	var coordPayload []byte
	switch factorization {
	case FactorizationCategorical:
		coordPayload, err = encodeModelledSymbols(coords, models.Coordinate, models.CoordinateModels)
	case FactorizationBitPlane:
		coordPayload, err = encodeModelledSymbols(coordinateBitPlaneSymbols(coords, code.BitWidth), models.CoordinateBit, models.CoordinateBitModels)
	default:
		return Payloads{}, fmt.Errorf("mirage: unsupported factorization %d", factorization)
	}
	if err != nil {
		return Payloads{}, err
	}
	normPayload, err := encodeModelledSymbols(byteSymbols(code.Norms), models.Norm, models.NormModels)
	if err != nil {
		return Payloads{}, err
	}
	return Payloads{CCoords: coordPayload, CNorms: normPayload}, nil
}

// DecodeLatentPayloads decodes .mrg payload streams back into a TurboQuant
// latent code using default factorized models.
func DecodeLatentPayloads(payloads Payloads, shape LatentShape, bitWidth int, factorization Factorization) (LatentCode, error) {
	models, err := DefaultLatentPayloadModels(bitWidth)
	if err != nil {
		return LatentCode{}, err
	}
	return DecodeLatentPayloadsWithModels(payloads, shape, bitWidth, factorization, models)
}

// DecodeLatentPayloadsWithModels mirrors EncodeLatentPayloadsWithModels.
func DecodeLatentPayloadsWithModels(payloads Payloads, shape LatentShape, bitWidth int, factorization Factorization, models LatentPayloadModels) (LatentCode, error) {
	if err := shape.validate(); err != nil {
		return LatentCode{}, err
	}
	if err := validateMirageBitWidth(bitWidth); err != nil {
		return LatentCode{}, err
	}
	coordCount := shape.Elements()
	var coords []uint16
	var err error
	switch factorization {
	case FactorizationCategorical:
		coords, err = decodeModelledSymbols(payloads.CCoords, coordCount, models.Coordinate, models.CoordinateModels)
	case FactorizationBitPlane:
		bits, decodeErr := decodeModelledSymbols(payloads.CCoords, coordCount*bitWidth, models.CoordinateBit, models.CoordinateBitModels)
		if decodeErr != nil {
			return LatentCode{}, decodeErr
		}
		coords, err = coordinateSymbolsFromBitPlanes(bits, bitWidth)
	default:
		return LatentCode{}, fmt.Errorf("mirage: unsupported factorization %d", factorization)
	}
	if err != nil {
		return LatentCode{}, err
	}
	normSymbols, err := decodeModelledSymbols(payloads.CNorms, shape.Positions(), models.Norm, models.NormModels)
	if err != nil {
		return LatentCode{}, err
	}
	normBytes, err := symbolsToBytes(normSymbols)
	if err != nil {
		return LatentCode{}, err
	}
	packed, err := PackCoordinateSymbols(coords, shape, bitWidth)
	if err != nil {
		return LatentCode{}, err
	}
	code := LatentCode{
		Shape:       shape,
		BitWidth:    bitWidth,
		Coordinates: packed,
		Norms:       normBytes,
	}
	return code, code.Validate()
}

func encodeModelledSymbols(symbols []uint16, model CDF, models []CDF) ([]byte, error) {
	if len(models) > 0 {
		return EncodeSymbolsWithModels(symbols, models)
	}
	return EncodeSymbols(symbols, model)
}

func decodeModelledSymbols(data []byte, count int, model CDF, models []CDF) ([]uint16, error) {
	if len(models) > 0 {
		if len(models) != count {
			return nil, fmt.Errorf("mirage: model count %d does not match symbol count %d", len(models), count)
		}
		return DecodeSymbolsWithModels(data, models)
	}
	return DecodeSymbols(data, count, model)
}

// UnpackCoordinateSymbols expands packed TurboQuant coordinate bytes into one
// integer symbol per latent coordinate.
func UnpackCoordinateSymbols(code LatentCode) ([]uint16, error) {
	if err := code.Validate(); err != nil {
		return nil, err
	}
	out := make([]uint16, 0, code.Shape.Elements())
	perPosition := code.PackedBytesPerPosition()
	tmp := make([]uint16, code.Shape.Channels)
	for pos := 0; pos < code.Shape.Positions(); pos++ {
		src := code.Coordinates[pos*perPosition : (pos+1)*perPosition]
		unpackCoordinateBlock(tmp, src, code.BitWidth)
		out = append(out, tmp...)
	}
	return out, nil
}

// PackCoordinateSymbols packs one integer symbol per latent coordinate into the
// TurboQuant coordinate byte layout.
func PackCoordinateSymbols(symbols []uint16, shape LatentShape, bitWidth int) ([]byte, error) {
	if err := shape.validate(); err != nil {
		return nil, err
	}
	if err := validateMirageBitWidth(bitWidth); err != nil {
		return nil, err
	}
	if len(symbols) != shape.Elements() {
		return nil, fmt.Errorf("mirage: coordinate symbol length %d does not match shape %d", len(symbols), shape.Elements())
	}
	limit := uint16(1 << bitWidth)
	perPosition := turboquant.PackedSize(shape.Channels, bitWidth)
	out := make([]byte, shape.Positions()*perPosition)
	for pos := 0; pos < shape.Positions(); pos++ {
		block := symbols[pos*shape.Channels : (pos+1)*shape.Channels]
		for _, sym := range block {
			if sym >= limit {
				return nil, fmt.Errorf("mirage: coordinate symbol %d exceeds bit width %d", sym, bitWidth)
			}
		}
		packCoordinateBlock(out[pos*perPosition:(pos+1)*perPosition], block, bitWidth)
	}
	return out, nil
}

func byteSymbols(data []byte) []uint16 {
	out := make([]uint16, len(data))
	for i, b := range data {
		out[i] = uint16(b)
	}
	return out
}

func symbolsToBytes(symbols []uint16) ([]byte, error) {
	out := make([]byte, len(symbols))
	for i, sym := range symbols {
		if sym > 255 {
			return nil, fmt.Errorf("mirage: q_norm symbol %d exceeds byte range at %d", sym, i)
		}
		out[i] = byte(sym)
	}
	return out, nil
}

func coordinateBitPlaneSymbols(symbols []uint16, bitWidth int) []uint16 {
	out := make([]uint16, 0, len(symbols)*bitWidth)
	for _, sym := range symbols {
		for bit := bitWidth - 1; bit >= 0; bit-- {
			out = append(out, (sym>>uint(bit))&1)
		}
	}
	return out
}

func coordinateSymbolsFromBitPlanes(bits []uint16, bitWidth int) ([]uint16, error) {
	if bitWidth <= 0 || len(bits)%bitWidth != 0 {
		return nil, fmt.Errorf("mirage: bit-plane symbol length %d does not divide bit width %d", len(bits), bitWidth)
	}
	out := make([]uint16, len(bits)/bitWidth)
	for i := range out {
		var sym uint16
		for bit := 0; bit < bitWidth; bit++ {
			v := bits[i*bitWidth+bit]
			if v > 1 {
				return nil, fmt.Errorf("mirage: invalid bit-plane symbol %d", v)
			}
			sym = (sym << 1) | v
		}
		out[i] = sym
	}
	return out, nil
}

func packCoordinateBlock(dst []byte, symbols []uint16, bitWidth int) {
	for i := range dst {
		dst[i] = 0
	}
	switch bitWidth {
	case 2:
		for i, sym := range symbols {
			dst[i/4] |= byte(sym&3) << uint((i%4)*2)
		}
	case 4:
		for i, sym := range symbols {
			dst[i/2] |= byte(sym&15) << uint((i%2)*4)
		}
	case 8:
		for i, sym := range symbols {
			dst[i] = byte(sym)
		}
	}
}

func unpackCoordinateBlock(dst []uint16, src []byte, bitWidth int) {
	switch bitWidth {
	case 2:
		for i := range dst {
			dst[i] = uint16((src[i/4] >> uint((i%4)*2)) & 3)
		}
	case 4:
		for i := range dst {
			dst[i] = uint16((src[i/2] >> uint((i%2)*4)) & 15)
		}
	case 8:
		for i := range dst {
			dst[i] = uint16(src[i])
		}
	}
}
