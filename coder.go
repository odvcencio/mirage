package mirage

import (
	"fmt"
	"sort"
)

const maxCDFTotal uint32 = 1 << 16

const (
	codeMax      uint64 = 1<<32 - 1
	codeHalf     uint64 = 1 << 31
	codeQuarter  uint64 = 1 << 30
	code3Quarter uint64 = codeQuarter * 3
)

// CDF is a static cumulative distribution used by the v1 arithmetic coder.
// Counts may contain zero-probability symbols, but encoded symbols must have a
// non-zero count.
type CDF struct {
	Counts     []uint32
	Cumulative []uint32
	Total      uint32
}

// NewCDF builds a validated static model. Total count is capped to keep the
// 32-bit arithmetic coder numerically simple and deterministic across Go/WASM.
func NewCDF(counts []uint32) (CDF, error) {
	if len(counts) == 0 {
		return CDF{}, fmt.Errorf("mirage: empty CDF")
	}
	copied := append([]uint32(nil), counts...)
	cumulative := make([]uint32, len(copied)+1)
	var total uint64
	for i, count := range copied {
		total += uint64(count)
		if total > uint64(maxCDFTotal) {
			return CDF{}, fmt.Errorf("mirage: CDF total %d exceeds %d", total, maxCDFTotal)
		}
		cumulative[i+1] = uint32(total)
	}
	if total == 0 {
		return CDF{}, fmt.Errorf("mirage: CDF total must be non-zero")
	}
	return CDF{
		Counts:     copied,
		Cumulative: cumulative,
		Total:      uint32(total),
	}, nil
}

// UniformCDF returns a uniform model with one count per symbol.
func UniformCDF(symbols int) (CDF, error) {
	if symbols <= 0 {
		return CDF{}, fmt.Errorf("mirage: symbol count must be positive")
	}
	if symbols > int(maxCDFTotal) {
		return CDF{}, fmt.Errorf("mirage: symbol count %d exceeds %d", symbols, maxCDFTotal)
	}
	counts := make([]uint32, symbols)
	for i := range counts {
		counts[i] = 1
	}
	return NewCDF(counts)
}

// SymbolCount returns the number of symbols modeled by the CDF.
func (c CDF) SymbolCount() int {
	return len(c.Counts)
}

func (c CDF) validate() error {
	if len(c.Counts) == 0 || len(c.Cumulative) != len(c.Counts)+1 || c.Total == 0 {
		return fmt.Errorf("mirage: invalid CDF")
	}
	if c.Total > maxCDFTotal {
		return fmt.Errorf("mirage: CDF total %d exceeds %d", c.Total, maxCDFTotal)
	}
	if c.Cumulative[0] != 0 || c.Cumulative[len(c.Cumulative)-1] != c.Total {
		return fmt.Errorf("mirage: invalid CDF cumulative bounds")
	}
	prev := c.Cumulative[0]
	for i := range c.Counts {
		next := c.Cumulative[i+1]
		if next < prev {
			return fmt.Errorf("mirage: CDF cumulative values decrease at symbol %d", i)
		}
		if next-prev != c.Counts[i] {
			return fmt.Errorf("mirage: inconsistent CDF count at symbol %d", i)
		}
		prev = next
	}
	return nil
}

// EncodeSymbols arithmetic-codes symbols with a static CDF.
func EncodeSymbols(symbols []uint16, model CDF) ([]byte, error) {
	if err := model.validate(); err != nil {
		return nil, err
	}
	if len(symbols) == 0 {
		return nil, nil
	}
	enc := arithmeticEncoder{
		low:  0,
		high: codeMax,
	}
	for _, sym16 := range symbols {
		sym := int(sym16)
		if sym < 0 || sym >= len(model.Counts) {
			return nil, fmt.Errorf("mirage: symbol %d outside CDF alphabet %d", sym, len(model.Counts))
		}
		if model.Counts[sym] == 0 {
			return nil, fmt.Errorf("mirage: symbol %d has zero probability", sym)
		}
		enc.encode(uint64(model.Cumulative[sym]), uint64(model.Cumulative[sym+1]), uint64(model.Total))
	}
	return enc.finish(), nil
}

// DecodeSymbols decodes exactly count symbols from data with a static CDF.
func DecodeSymbols(data []byte, count int, model CDF) ([]uint16, error) {
	if err := model.validate(); err != nil {
		return nil, err
	}
	if count < 0 {
		return nil, fmt.Errorf("mirage: negative symbol count %d", count)
	}
	if count == 0 {
		return nil, nil
	}
	dec := newArithmeticDecoder(data)
	out := make([]uint16, count)
	for i := range out {
		sym, err := dec.decode(model)
		if err != nil {
			return nil, err
		}
		out[i] = uint16(sym)
	}
	return out, nil
}

// EncodeSymbolsWithModels arithmetic-codes symbols with one CDF per symbol.
// This is the host-side boundary used by learned entropy models whose
// probabilities vary by latent coordinate.
func EncodeSymbolsWithModels(symbols []uint16, models []CDF) ([]byte, error) {
	if len(symbols) != len(models) {
		return nil, fmt.Errorf("mirage: symbol/model length mismatch %d != %d", len(symbols), len(models))
	}
	if len(symbols) == 0 {
		return nil, nil
	}
	enc := arithmeticEncoder{
		low:  0,
		high: codeMax,
	}
	for i, sym16 := range symbols {
		model := models[i]
		if err := model.validate(); err != nil {
			return nil, fmt.Errorf("mirage: model %d: %w", i, err)
		}
		sym := int(sym16)
		if sym < 0 || sym >= len(model.Counts) {
			return nil, fmt.Errorf("mirage: symbol %d outside CDF alphabet %d at %d", sym, len(model.Counts), i)
		}
		if model.Counts[sym] == 0 {
			return nil, fmt.Errorf("mirage: symbol %d has zero probability at %d", sym, i)
		}
		enc.encode(uint64(model.Cumulative[sym]), uint64(model.Cumulative[sym+1]), uint64(model.Total))
	}
	return enc.finish(), nil
}

// DecodeSymbolsWithModels decodes exactly len(models) symbols with one CDF per
// symbol. The model sequence must match the sequence used during encode.
func DecodeSymbolsWithModels(data []byte, models []CDF) ([]uint16, error) {
	if len(models) == 0 {
		return nil, nil
	}
	dec := newArithmeticDecoder(data)
	out := make([]uint16, len(models))
	for i, model := range models {
		if err := model.validate(); err != nil {
			return nil, fmt.Errorf("mirage: model %d: %w", i, err)
		}
		sym, err := dec.decode(model)
		if err != nil {
			return nil, err
		}
		out[i] = uint16(sym)
	}
	return out, nil
}

// EncodeBytes arithmetic-codes bytes with a uniform byte model. This is mainly
// useful for smoke tests and payloads whose probability model is external.
func EncodeBytes(data []byte) ([]byte, error) {
	model, err := UniformCDF(256)
	if err != nil {
		return nil, err
	}
	symbols := make([]uint16, len(data))
	for i, b := range data {
		symbols[i] = uint16(b)
	}
	return EncodeSymbols(symbols, model)
}

// DecodeBytes decodes exactly count bytes encoded by EncodeBytes.
func DecodeBytes(data []byte, count int) ([]byte, error) {
	model, err := UniformCDF(256)
	if err != nil {
		return nil, err
	}
	symbols, err := DecodeSymbols(data, count, model)
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(symbols))
	for i, sym := range symbols {
		out[i] = byte(sym)
	}
	return out, nil
}

type arithmeticEncoder struct {
	low     uint64
	high    uint64
	pending int
	bits    bitWriter
}

func (e *arithmeticEncoder) encode(cumLow, cumHigh, total uint64) {
	width := e.high - e.low + 1
	e.high = e.low + (width*cumHigh)/total - 1
	e.low = e.low + (width*cumLow)/total
	for {
		switch {
		case e.high < codeHalf:
			e.emitBit(0)
		case e.low >= codeHalf:
			e.emitBit(1)
			e.low -= codeHalf
			e.high -= codeHalf
		case e.low >= codeQuarter && e.high < code3Quarter:
			e.pending++
			e.low -= codeQuarter
			e.high -= codeQuarter
		default:
			return
		}
		e.low <<= 1
		e.high = (e.high << 1) | 1
	}
}

func (e *arithmeticEncoder) emitBit(bit uint8) {
	e.bits.writeBit(bit)
	for e.pending > 0 {
		e.bits.writeBit(bit ^ 1)
		e.pending--
	}
}

func (e *arithmeticEncoder) finish() []byte {
	e.pending++
	if e.low < codeQuarter {
		e.emitBit(0)
	} else {
		e.emitBit(1)
	}
	return e.bits.bytes()
}

type arithmeticDecoder struct {
	low   uint64
	high  uint64
	value uint64
	bits  bitReader
}

func newArithmeticDecoder(data []byte) *arithmeticDecoder {
	d := &arithmeticDecoder{
		low:  0,
		high: codeMax,
		bits: bitReader{data: data},
	}
	for i := 0; i < 32; i++ {
		d.value = (d.value << 1) | uint64(d.bits.readBit())
	}
	return d
}

func (d *arithmeticDecoder) decode(model CDF) (int, error) {
	if d.value < d.low || d.value > d.high {
		return 0, fmt.Errorf("mirage: arithmetic stream state is outside active interval")
	}
	width := d.high - d.low + 1
	scaled := ((d.value-d.low+1)*uint64(model.Total) - 1) / width
	if scaled >= uint64(model.Total) {
		return 0, fmt.Errorf("mirage: arithmetic stream is outside model range")
	}
	sym := findCDFSymbol(model.Cumulative, uint32(scaled))
	if sym < 0 || sym >= len(model.Counts) || model.Counts[sym] == 0 {
		return 0, fmt.Errorf("mirage: arithmetic stream selected invalid symbol")
	}
	cumLow := uint64(model.Cumulative[sym])
	cumHigh := uint64(model.Cumulative[sym+1])
	d.high = d.low + (width*cumHigh)/uint64(model.Total) - 1
	d.low = d.low + (width*cumLow)/uint64(model.Total)
	for {
		switch {
		case d.high < codeHalf:
		case d.low >= codeHalf:
			d.value -= codeHalf
			d.low -= codeHalf
			d.high -= codeHalf
		case d.low >= codeQuarter && d.high < code3Quarter:
			d.value -= codeQuarter
			d.low -= codeQuarter
			d.high -= codeQuarter
		default:
			return sym, nil
		}
		d.low <<= 1
		d.high = (d.high << 1) | 1
		d.value = (d.value << 1) | uint64(d.bits.readBit())
	}
}

func findCDFSymbol(cumulative []uint32, scaled uint32) int {
	i := sort.Search(len(cumulative)-1, func(i int) bool {
		return cumulative[i+1] > scaled
	})
	if i >= len(cumulative)-1 || cumulative[i] > scaled {
		return -1
	}
	return i
}

type bitWriter struct {
	data []byte
	cur  byte
	used uint8
}

func (w *bitWriter) writeBit(bit uint8) {
	if bit != 0 {
		w.cur |= 1 << (7 - w.used)
	}
	w.used++
	if w.used == 8 {
		w.data = append(w.data, w.cur)
		w.cur = 0
		w.used = 0
	}
}

func (w *bitWriter) bytes() []byte {
	out := append([]byte(nil), w.data...)
	if w.used != 0 {
		out = append(out, w.cur)
	}
	return out
}

type bitReader struct {
	data []byte
	pos  int
	used uint8
}

func (r *bitReader) readBit() uint8 {
	if r.pos >= len(r.data) {
		return 0
	}
	bit := (r.data[r.pos] >> (7 - r.used)) & 1
	r.used++
	if r.used == 8 {
		r.used = 0
		r.pos++
	}
	return bit
}
