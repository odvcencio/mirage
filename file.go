package mirage

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
)

// Payloads are the three arithmetic-coded streams carried by a .mrg v1 file.
type Payloads struct {
	CZ      []byte
	CCoords []byte
	CNorms  []byte
}

// File is a parsed Mirage v1 bitstream.
type File struct {
	Header   Header
	Payloads Payloads
}

// BuildFile constructs a validated File and fills the header payload lengths.
func BuildFile(opts HeaderOptions, payloads Payloads) (File, error) {
	czBytes, err := lenUint32("c_z", len(payloads.CZ))
	if err != nil {
		return File{}, err
	}
	cCoordsBytes, err := lenUint32("c_coords", len(payloads.CCoords))
	if err != nil {
		return File{}, err
	}
	cNormsBytes, err := lenUint32("c_norms", len(payloads.CNorms))
	if err != nil {
		return File{}, err
	}
	opts.CZBytes = czBytes
	opts.CCoordsBytes = cCoordsBytes
	opts.CNormsBytes = cNormsBytes
	h, err := NewHeader(opts)
	if err != nil {
		return File{}, err
	}
	return File{
		Header: h,
		Payloads: Payloads{
			CZ:      append([]byte(nil), payloads.CZ...),
			CCoords: append([]byte(nil), payloads.CCoords...),
			CNorms:  append([]byte(nil), payloads.CNorms...),
		},
	}, nil
}

// MarshalBinary serializes a complete .mrg file including trailing CRC-32.
func (f File) MarshalBinary() ([]byte, error) {
	if err := f.validatePayloadLengths(); err != nil {
		return nil, err
	}
	header, err := f.Header.MarshalBinary()
	if err != nil {
		return nil, err
	}
	size := int(f.Header.TotalBytes())
	out := make([]byte, 0, size)
	out = append(out, header...)
	out = append(out, f.Payloads.CZ...)
	out = append(out, f.Payloads.CCoords...)
	out = append(out, f.Payloads.CNorms...)
	out = appendCRC32(out)
	return out, nil
}

// WriteTo writes a complete .mrg file to w.
func (f File) WriteTo(w io.Writer) (int64, error) {
	data, err := f.MarshalBinary()
	if err != nil {
		return 0, err
	}
	n, err := w.Write(data)
	return int64(n), err
}

// ParseFile parses and validates a complete .mrg file.
func ParseFile(data []byte) (File, error) {
	if len(data) < HeaderSize+CRCSize {
		return File{}, fmt.Errorf("mirage: file too short: got %d bytes", len(data))
	}
	h, err := ParseHeader(data[:HeaderSize])
	if err != nil {
		return File{}, err
	}
	if uint64(len(data)) != h.TotalBytes() {
		return File{}, fmt.Errorf("mirage: file length %d does not match header total %d", len(data), h.TotalBytes())
	}
	wantCRC := binary.LittleEndian.Uint32(data[len(data)-CRCSize:])
	gotCRC := crc32.ChecksumIEEE(data[:len(data)-CRCSize])
	if gotCRC != wantCRC {
		return File{}, fmt.Errorf("mirage: crc32 mismatch")
	}
	pos := HeaderSize
	czEnd := pos + int(h.CZBytes)
	coordsEnd := czEnd + int(h.CCoordsBytes)
	normsEnd := coordsEnd + int(h.CNormsBytes)
	return File{
		Header: h,
		Payloads: Payloads{
			CZ:      append([]byte(nil), data[pos:czEnd]...),
			CCoords: append([]byte(nil), data[czEnd:coordsEnd]...),
			CNorms:  append([]byte(nil), data[coordsEnd:normsEnd]...),
		},
	}, nil
}

// ReadFile reads, parses, and validates a complete .mrg file.
func ReadFile(r io.Reader) (File, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return File{}, err
	}
	return ParseFile(data)
}

func (f File) validatePayloadLengths() error {
	if err := f.Header.Validate(); err != nil {
		return err
	}
	if uint32(len(f.Payloads.CZ)) != f.Header.CZBytes {
		return fmt.Errorf("mirage: c_z length %d does not match header %d", len(f.Payloads.CZ), f.Header.CZBytes)
	}
	if uint32(len(f.Payloads.CCoords)) != f.Header.CCoordsBytes {
		return fmt.Errorf("mirage: c_coords length %d does not match header %d", len(f.Payloads.CCoords), f.Header.CCoordsBytes)
	}
	if uint32(len(f.Payloads.CNorms)) != f.Header.CNormsBytes {
		return fmt.Errorf("mirage: c_norms length %d does not match header %d", len(f.Payloads.CNorms), f.Header.CNormsBytes)
	}
	return nil
}

func lenUint32(name string, n int) (uint32, error) {
	if n < 0 {
		return 0, fmt.Errorf("mirage: %s length is negative", name)
	}
	if uint64(n) > uint64(^uint32(0)) {
		return 0, fmt.Errorf("mirage: %s length %d exceeds uint32", name, n)
	}
	return uint32(n), nil
}
