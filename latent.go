package mirage

import (
	"fmt"
	"math"

	turboquant "github.com/odvcencio/turboquant"
)

// LatentShape is the channel-first shape of one Mirage latent tensor.
type LatentShape struct {
	Channels int
	Height   int
	Width    int
}

// Positions returns Height*Width.
func (s LatentShape) Positions() int {
	return s.Height * s.Width
}

// Elements returns Channels*Height*Width.
func (s LatentShape) Elements() int {
	return s.Channels * s.Positions()
}

func (s LatentShape) validate() error {
	if s.Channels < 2 {
		return fmt.Errorf("mirage: latent channels must be >= 2")
	}
	if s.Height <= 0 || s.Width <= 0 {
		return fmt.Errorf("mirage: latent height and width must be positive")
	}
	return nil
}

func (s LatentShape) offset(c, y, x int) int {
	return c*s.Height*s.Width + y*s.Width + x
}

// LatentCode is the TurboQuant codeword stream and q_norm stream for a Mirage
// latent tensor. Coordinates are concatenated per spatial position.
type LatentCode struct {
	Shape       LatentShape
	BitWidth    int
	Coordinates []byte
	Norms       []byte
}

// PackedBytesPerPosition returns the number of coordinate bytes per H/W slot.
func (c LatentCode) PackedBytesPerPosition() int {
	return turboquant.PackedSize(c.Shape.Channels, c.BitWidth)
}

// Validate checks that the code buffers match their shape and bit width.
func (c LatentCode) Validate() error {
	if err := c.Shape.validate(); err != nil {
		return err
	}
	if err := validateMirageBitWidth(c.BitWidth); err != nil {
		return err
	}
	wantCoords := c.Shape.Positions() * c.PackedBytesPerPosition()
	if len(c.Coordinates) != wantCoords {
		return fmt.Errorf("mirage: coordinate length %d does not match shape %d", len(c.Coordinates), wantCoords)
	}
	if len(c.Norms) != c.Shape.Positions() {
		return fmt.Errorf("mirage: norm length %d does not match positions %d", len(c.Norms), c.Shape.Positions())
	}
	return nil
}

// EncodeLatents applies the structural TurboQuant codec to a CHW latent tensor.
// Each spatial position is quantized as one vector over channels, producing one
// q_norm byte per position.
func EncodeLatents(latents []float32, shape LatentShape, bitWidth int, seed int64) (LatentCode, error) {
	if err := shape.validate(); err != nil {
		return LatentCode{}, err
	}
	if err := validateMirageBitWidth(bitWidth); err != nil {
		return LatentCode{}, err
	}
	if len(latents) != shape.Elements() {
		return LatentCode{}, fmt.Errorf("mirage: latent length %d does not match shape %d", len(latents), shape.Elements())
	}
	for i, v := range latents {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return LatentCode{}, fmt.Errorf("mirage: invalid latent value at %d", i)
		}
	}
	q := turboquant.NewHadamardWithSeed(shape.Channels, bitWidth, seed)
	packedPerPosition := turboquant.PackedSize(shape.Channels, bitWidth)
	code := LatentCode{
		Shape:       shape,
		BitWidth:    bitWidth,
		Coordinates: make([]byte, shape.Positions()*packedPerPosition),
		Norms:       make([]byte, shape.Positions()),
	}
	vec := make([]float32, shape.Channels)
	for y := 0; y < shape.Height; y++ {
		for x := 0; x < shape.Width; x++ {
			pos := y*shape.Width + x
			for c := 0; c < shape.Channels; c++ {
				vec[c] = latents[shape.offset(c, y, x)]
			}
			dst := code.Coordinates[pos*packedPerPosition : (pos+1)*packedPerPosition]
			norm := q.QuantizeTo(dst, vec)
			code.Norms[pos] = QuantizeNorm(norm)
		}
	}
	return code, nil
}

// DecodeLatents reconstructs an approximate CHW latent tensor from a LatentCode.
func DecodeLatents(code LatentCode, seed int64) ([]float32, error) {
	if err := code.Validate(); err != nil {
		return nil, err
	}
	q := turboquant.NewHadamardWithSeed(code.Shape.Channels, code.BitWidth, seed)
	out := make([]float32, code.Shape.Elements())
	vec := make([]float32, code.Shape.Channels)
	packedPerPosition := code.PackedBytesPerPosition()
	for y := 0; y < code.Shape.Height; y++ {
		for x := 0; x < code.Shape.Width; x++ {
			pos := y*code.Shape.Width + x
			src := code.Coordinates[pos*packedPerPosition : (pos+1)*packedPerPosition]
			q.DequantizeTo(vec, src)
			norm := DequantizeNorm(code.Norms[pos])
			for c := 0; c < code.Shape.Channels; c++ {
				out[code.Shape.offset(c, y, x)] = vec[c] * norm
			}
		}
	}
	return out, nil
}
