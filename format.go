package mirage

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash/crc32"
)

const (
	// HeaderSize is the fixed Mirage v1 header size in bytes.
	HeaderSize = 72
	// CRCSize is the trailing CRC-32 size in bytes.
	CRCSize = 4
	// FormatVersion is the Mirage Image v1 format version.
	FormatVersion = 1
)

const (
	flagDistortionMSSSIM byte = 1 << 0
	flagBitPlane         byte = 1 << 1
	reservedFlagsMask    byte = ^byte(flagDistortionMSSSIM | flagBitPlane)
)

var magicMRG1 = [4]byte{'M', 'R', 'G', '1'}

// DistortionMetric records which loss family trained the model that produced
// a stream. It is metadata for decoding and reporting; it is not part of the
// entropy model itself.
type DistortionMetric uint8

const (
	DistortionMSE DistortionMetric = iota
	DistortionMSSSIM
)

func (m DistortionMetric) String() string {
	switch m {
	case DistortionMSE:
		return "mse"
	case DistortionMSSSIM:
		return "ms-ssim"
	default:
		return fmt.Sprintf("distortion(%d)", m)
	}
}

// Factorization records the coordinate entropy-model shape used by the file.
type Factorization uint8

const (
	FactorizationCategorical Factorization = iota
	FactorizationBitPlane
)

func (f Factorization) String() string {
	switch f {
	case FactorizationCategorical:
		return "categorical"
	case FactorizationBitPlane:
		return "bit-plane"
	default:
		return fmt.Sprintf("factorization(%d)", f)
	}
}

// Header is the fixed 72-byte .mrg v1 header without the magic/version fields.
// Multi-byte fields are little-endian on the wire.
type Header struct {
	Flags            byte
	BitWidth         uint8
	ImageWidth       uint32
	ImageHeight      uint32
	LatentChannels   uint32
	LatentHeight     uint32
	LatentWidth      uint32
	ModelFingerprint [32]byte
	CZBytes          uint32
	CCoordsBytes     uint32
	CNormsBytes      uint32
}

// HeaderOptions describes the caller-facing construction surface for a v1
// header. Payload byte counts may be left at zero and filled by BuildFile.
type HeaderOptions struct {
	DistortionMetric DistortionMetric
	Factorization    Factorization
	BitWidth         int
	ImageWidth       uint32
	ImageHeight      uint32
	LatentChannels   uint32
	LatentHeight     uint32
	LatentWidth      uint32
	ModelFingerprint [32]byte
	CZBytes          uint32
	CCoordsBytes     uint32
	CNormsBytes      uint32
}

// FingerprintModel returns the SHA-256 fingerprint stored in .mrg headers.
func FingerprintModel(modelArtifact []byte) [32]byte {
	return sha256.Sum256(modelArtifact)
}

// NewHeader builds and validates a Mirage v1 header.
func NewHeader(opts HeaderOptions) (Header, error) {
	if err := validateMirageBitWidth(opts.BitWidth); err != nil {
		return Header{}, err
	}
	var flags byte
	switch opts.DistortionMetric {
	case DistortionMSE:
	case DistortionMSSSIM:
		flags |= flagDistortionMSSSIM
	default:
		return Header{}, fmt.Errorf("mirage: unsupported distortion metric %d", opts.DistortionMetric)
	}
	switch opts.Factorization {
	case FactorizationCategorical:
	case FactorizationBitPlane:
		flags |= flagBitPlane
	default:
		return Header{}, fmt.Errorf("mirage: unsupported factorization %d", opts.Factorization)
	}
	h := Header{
		Flags:            flags,
		BitWidth:         uint8(opts.BitWidth),
		ImageWidth:       opts.ImageWidth,
		ImageHeight:      opts.ImageHeight,
		LatentChannels:   opts.LatentChannels,
		LatentHeight:     opts.LatentHeight,
		LatentWidth:      opts.LatentWidth,
		ModelFingerprint: opts.ModelFingerprint,
		CZBytes:          opts.CZBytes,
		CCoordsBytes:     opts.CCoordsBytes,
		CNormsBytes:      opts.CNormsBytes,
	}
	if err := h.Validate(); err != nil {
		return Header{}, err
	}
	return h, nil
}

// Validate checks the fields whose invariants are fixed by the v1 spec.
func (h Header) Validate() error {
	if h.Flags&reservedFlagsMask != 0 {
		return fmt.Errorf("mirage: reserved flag bits set: 0x%02x", h.Flags&reservedFlagsMask)
	}
	if err := validateMirageBitWidth(int(h.BitWidth)); err != nil {
		return err
	}
	if h.ImageWidth == 0 || h.ImageHeight == 0 {
		return fmt.Errorf("mirage: image dimensions must be non-zero")
	}
	if h.LatentChannels == 0 || h.LatentHeight == 0 || h.LatentWidth == 0 {
		return fmt.Errorf("mirage: latent dimensions must be non-zero")
	}
	return nil
}

// DistortionMetric decodes the v1 flags distortion bit.
func (h Header) DistortionMetric() DistortionMetric {
	if h.Flags&flagDistortionMSSSIM != 0 {
		return DistortionMSSSIM
	}
	return DistortionMSE
}

// Factorization decodes the v1 flags factorization bit.
func (h Header) Factorization() Factorization {
	if h.Flags&flagBitPlane != 0 {
		return FactorizationBitPlane
	}
	return FactorizationCategorical
}

// LatentShape returns the main-latent shape described by the header.
func (h Header) LatentShape() LatentShape {
	return LatentShape{
		Channels: int(h.LatentChannels),
		Height:   int(h.LatentHeight),
		Width:    int(h.LatentWidth),
	}
}

// ImagePixels returns ImageWidth*ImageHeight as uint64.
func (h Header) ImagePixels() uint64 {
	return uint64(h.ImageWidth) * uint64(h.ImageHeight)
}

// PayloadBytes returns the total encoded payload byte count.
func (h Header) PayloadBytes() uint64 {
	return uint64(h.CZBytes) + uint64(h.CCoordsBytes) + uint64(h.CNormsBytes)
}

// TotalBytes returns the complete .mrg file size including fixed header and CRC.
func (h Header) TotalBytes() uint64 {
	return HeaderSize + h.PayloadBytes() + CRCSize
}

// MarshalBinary returns the exact 72-byte fixed header.
func (h Header) MarshalBinary() ([]byte, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	buf := make([]byte, HeaderSize)
	copy(buf[0:4], magicMRG1[:])
	buf[4] = FormatVersion
	buf[5] = h.Flags
	buf[6] = h.BitWidth
	buf[7] = 0
	binary.LittleEndian.PutUint32(buf[8:12], h.ImageWidth)
	binary.LittleEndian.PutUint32(buf[12:16], h.ImageHeight)
	binary.LittleEndian.PutUint32(buf[16:20], h.LatentChannels)
	binary.LittleEndian.PutUint32(buf[20:24], h.LatentHeight)
	binary.LittleEndian.PutUint32(buf[24:28], h.LatentWidth)
	copy(buf[28:60], h.ModelFingerprint[:])
	binary.LittleEndian.PutUint32(buf[60:64], h.CZBytes)
	binary.LittleEndian.PutUint32(buf[64:68], h.CCoordsBytes)
	binary.LittleEndian.PutUint32(buf[68:72], h.CNormsBytes)
	return buf, nil
}

// ParseHeader parses and validates the fixed 72-byte .mrg header.
func ParseHeader(data []byte) (Header, error) {
	if len(data) < HeaderSize {
		return Header{}, fmt.Errorf("mirage: header too short: got %d bytes want %d", len(data), HeaderSize)
	}
	if string(data[0:4]) != string(magicMRG1[:]) {
		return Header{}, fmt.Errorf("mirage: invalid magic %q", string(data[0:4]))
	}
	if data[4] != FormatVersion {
		return Header{}, fmt.Errorf("mirage: unsupported format version %d", data[4])
	}
	if data[7] != 0 {
		return Header{}, fmt.Errorf("mirage: reserved byte must be zero")
	}
	h := Header{
		Flags:          data[5],
		BitWidth:       data[6],
		ImageWidth:     binary.LittleEndian.Uint32(data[8:12]),
		ImageHeight:    binary.LittleEndian.Uint32(data[12:16]),
		LatentChannels: binary.LittleEndian.Uint32(data[16:20]),
		LatentHeight:   binary.LittleEndian.Uint32(data[20:24]),
		LatentWidth:    binary.LittleEndian.Uint32(data[24:28]),
		CZBytes:        binary.LittleEndian.Uint32(data[60:64]),
		CCoordsBytes:   binary.LittleEndian.Uint32(data[64:68]),
		CNormsBytes:    binary.LittleEndian.Uint32(data[68:72]),
	}
	copy(h.ModelFingerprint[:], data[28:60])
	if err := h.Validate(); err != nil {
		return Header{}, err
	}
	return h, nil
}

func appendCRC32(data []byte) []byte {
	crc := crc32.ChecksumIEEE(data)
	return binary.LittleEndian.AppendUint32(data, crc)
}

func validateMirageBitWidth(bitWidth int) error {
	switch bitWidth {
	case 2, 4, 8:
		return nil
	default:
		return fmt.Errorf("mirage: bit width must be 2, 4, or 8, got %d", bitWidth)
	}
}
