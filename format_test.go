package mirage

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func testHeaderOptions() HeaderOptions {
	return HeaderOptions{
		DistortionMetric: DistortionMSSSIM,
		Factorization:    FactorizationBitPlane,
		BitWidth:         4,
		ImageWidth:       768,
		ImageHeight:      512,
		LatentChannels:   192,
		LatentHeight:     32,
		LatentWidth:      48,
		ModelFingerprint: FingerprintModel([]byte("model")),
		CZBytes:          3,
		CCoordsBytes:     4,
		CNormsBytes:      5,
	}
}

func TestHeaderMarshalMatchesSpecOffsets(t *testing.T) {
	h, err := NewHeader(testHeaderOptions())
	if err != nil {
		t.Fatalf("NewHeader: %v", err)
	}
	data, err := h.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	if len(data) != HeaderSize {
		t.Fatalf("header length = %d want %d", len(data), HeaderSize)
	}
	if string(data[0:4]) != "MRG1" {
		t.Fatalf("magic = %q", data[0:4])
	}
	if data[4] != FormatVersion {
		t.Fatalf("format version = %d want %d", data[4], FormatVersion)
	}
	if data[5] != flagDistortionMSSSIM|flagBitPlane {
		t.Fatalf("flags = 0x%02x", data[5])
	}
	if data[6] != 4 || data[7] != 0 {
		t.Fatalf("bit width/reserved = %d/%d", data[6], data[7])
	}
	if got := binary.LittleEndian.Uint32(data[8:12]); got != 768 {
		t.Fatalf("image width = %d", got)
	}
	if got := binary.LittleEndian.Uint32(data[12:16]); got != 512 {
		t.Fatalf("image height = %d", got)
	}
	if got := binary.LittleEndian.Uint32(data[16:20]); got != 192 {
		t.Fatalf("latent channels = %d", got)
	}
	if got := binary.LittleEndian.Uint32(data[20:24]); got != 32 {
		t.Fatalf("latent height = %d", got)
	}
	if got := binary.LittleEndian.Uint32(data[24:28]); got != 48 {
		t.Fatalf("latent width = %d", got)
	}
	if !bytes.Equal(data[28:60], h.ModelFingerprint[:]) {
		t.Fatalf("fingerprint did not round-trip")
	}
	if got := binary.LittleEndian.Uint32(data[60:64]); got != 3 {
		t.Fatalf("c_z bytes = %d", got)
	}
	if got := binary.LittleEndian.Uint32(data[64:68]); got != 4 {
		t.Fatalf("c_coords bytes = %d", got)
	}
	if got := binary.LittleEndian.Uint32(data[68:72]); got != 5 {
		t.Fatalf("c_norms bytes = %d", got)
	}
}

func TestHeaderParseRoundTrip(t *testing.T) {
	h, err := NewHeader(testHeaderOptions())
	if err != nil {
		t.Fatalf("NewHeader: %v", err)
	}
	data, err := h.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	got, err := ParseHeader(data)
	if err != nil {
		t.Fatalf("ParseHeader: %v", err)
	}
	if got != h {
		t.Fatalf("header round-trip mismatch\n got %#v\nwant %#v", got, h)
	}
	if got.DistortionMetric() != DistortionMSSSIM {
		t.Fatalf("distortion = %v", got.DistortionMetric())
	}
	if got.Factorization() != FactorizationBitPlane {
		t.Fatalf("factorization = %v", got.Factorization())
	}
	if got.TotalBytes() != HeaderSize+3+4+5+CRCSize {
		t.Fatalf("total bytes = %d", got.TotalBytes())
	}
}

func TestHeaderRejectsReservedFields(t *testing.T) {
	h, err := NewHeader(testHeaderOptions())
	if err != nil {
		t.Fatalf("NewHeader: %v", err)
	}
	data, err := h.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	data[5] |= 0x80
	if _, err := ParseHeader(data); err == nil {
		t.Fatalf("ParseHeader accepted reserved flag bit")
	}
	data[5] &^= 0x80
	data[7] = 1
	if _, err := ParseHeader(data); err == nil {
		t.Fatalf("ParseHeader accepted reserved byte")
	}
}

func TestFileBuildMarshalParse(t *testing.T) {
	opts := testHeaderOptions()
	payloads := Payloads{
		CZ:      []byte{1, 2, 3},
		CCoords: []byte{4, 5, 6, 7},
		CNorms:  []byte{8, 9, 10, 11, 12},
	}
	file, err := BuildFile(opts, payloads)
	if err != nil {
		t.Fatalf("BuildFile: %v", err)
	}
	data, err := file.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	if len(data) != int(file.Header.TotalBytes()) {
		t.Fatalf("file length = %d want %d", len(data), file.Header.TotalBytes())
	}
	parsed, err := ParseFile(data)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}
	if parsed.Header != file.Header {
		t.Fatalf("parsed header mismatch")
	}
	if !bytes.Equal(parsed.Payloads.CZ, payloads.CZ) ||
		!bytes.Equal(parsed.Payloads.CCoords, payloads.CCoords) ||
		!bytes.Equal(parsed.Payloads.CNorms, payloads.CNorms) {
		t.Fatalf("payloads did not round-trip")
	}
}

func TestFileRejectsCRCDrift(t *testing.T) {
	file, err := BuildFile(testHeaderOptions(), Payloads{
		CZ:      []byte{1, 2, 3},
		CCoords: []byte{4, 5, 6, 7},
		CNorms:  []byte{8, 9, 10, 11, 12},
	})
	if err != nil {
		t.Fatalf("BuildFile: %v", err)
	}
	data, err := file.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	data[HeaderSize] ^= 0xff
	if _, err := ParseFile(data); err == nil {
		t.Fatalf("ParseFile accepted corrupted payload")
	}
}
