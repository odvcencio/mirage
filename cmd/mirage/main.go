package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
	"github.com/odvcencio/mirage"
)

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "encode":
		err = runEncode(os.Args[2:])
	case "decode":
		err = runDecode(os.Args[2:])
	case "info":
		err = runInfo(os.Args[2:])
	case "eval":
		err = runEval(os.Args[2:])
	case "init-manta":
		err = runInitManta(os.Args[2:])
	case "check-manta":
		err = runCheckManta(os.Args[2:])
	default:
		usage(os.Stderr)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "mirage:", err)
		os.Exit(1)
	}
}

func usage(w io.Writer) {
	fmt.Fprintln(w, "usage:")
	fmt.Fprintln(w, "  mirage encode -in image.png -out image.mrg [-bits 2|4|8] [-factorization categorical|bit-plane]")
	fmt.Fprintln(w, "  mirage decode -in image.mrg -out image.png")
	fmt.Fprintln(w, "  mirage info   -in image.mrg")
	fmt.Fprintln(w, "  mirage eval   -source image.png -mrg image.mrg")
	fmt.Fprintln(w, "  mirage init-manta -out mirage_v1.mll [-bits 2|4|8]")
	fmt.Fprintln(w, "  mirage check-manta -in mirage_v1.mll [-entry train_step]")
}

func runEncode(args []string) error {
	fs := flag.NewFlagSet("encode", flag.ExitOnError)
	in := fs.String("in", "", "input PNG, JPEG, or PPM")
	out := fs.String("out", "", "output .mrg path")
	bits := fs.Int("bits", 4, "TurboQuant bits per latent coordinate: 2, 4, or 8")
	factorizationFlag := fs.String("factorization", "categorical", "coordinate entropy factorization: categorical or bit-plane")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *in == "" || *out == "" {
		return fmt.Errorf("encode requires -in and -out")
	}
	data, err := os.ReadFile(*in)
	if err != nil {
		return err
	}
	factorization, err := parseFactorization(*factorizationFlag)
	if err != nil {
		return err
	}
	encoded, format, err := mirage.EncodeImageReader(data, mirage.EncodeOptions{BitWidth: *bits, Factorization: factorization})
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, encoded, 0o644); err != nil {
		return err
	}
	file, err := mirage.ParseFile(encoded)
	if err != nil {
		return err
	}
	fmt.Printf("encoded %s -> %s\n", *in, *out)
	fmt.Printf("source_format=%s bits=%d size=%d bpp=%.4f\n", format, *bits, len(encoded), mirage.BitsPerPixel(file))
	return nil
}

func parseFactorization(value string) (mirage.Factorization, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "categorical", "cat":
		return mirage.FactorizationCategorical, nil
	case "bit-plane", "bitplane", "bit_plane", "bits":
		return mirage.FactorizationBitPlane, nil
	default:
		return 0, fmt.Errorf("unknown factorization %q", value)
	}
}

func runDecode(args []string) error {
	fs := flag.NewFlagSet("decode", flag.ExitOnError)
	in := fs.String("in", "", "input .mrg path")
	out := fs.String("out", "", "output PNG path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *in == "" || *out == "" {
		return fmt.Errorf("decode requires -in and -out")
	}
	data, err := os.ReadFile(*in)
	if err != nil {
		return err
	}
	img, err := mirage.DecodeBytesMRG(data, mirage.DefaultDecodeOptions())
	if err != nil {
		return err
	}
	f, err := os.Create(*out)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := mirage.EncodePNG(f, img); err != nil {
		return err
	}
	fmt.Printf("decoded %s -> %s\n", *in, *out)
	fmt.Printf("image=%dx%d\n", img.Width, img.Height)
	return nil
}

func runInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	in := fs.String("in", "", "input .mrg path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *in == "" {
		return fmt.Errorf("info requires -in")
	}
	data, err := os.ReadFile(*in)
	if err != nil {
		return err
	}
	file, err := mirage.ParseFile(data)
	if err != nil {
		return err
	}
	h := file.Header
	fmt.Printf("format_version=%d\n", mirage.FormatVersion)
	fmt.Printf("image=%dx%d\n", h.ImageWidth, h.ImageHeight)
	fmt.Printf("latent=%dx%dx%d\n", h.LatentChannels, h.LatentHeight, h.LatentWidth)
	fmt.Printf("bit_width=%d\n", h.BitWidth)
	fmt.Printf("distortion=%s\n", h.DistortionMetric())
	fmt.Printf("factorization=%s\n", h.Factorization())
	fmt.Printf("payload_bytes=%d c_z=%d c_coords=%d c_norms=%d\n", h.PayloadBytes(), h.CZBytes, h.CCoordsBytes, h.CNormsBytes)
	fmt.Printf("container_bytes=%d bpp=%.4f\n", h.TotalBytes(), mirage.BitsPerPixel(file))
	fmt.Printf("model_fingerprint=%s\n", hex.EncodeToString(h.ModelFingerprint[:]))
	return nil
}

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	sourcePath := fs.String("source", "", "source PNG, JPEG, or PPM")
	mrgPath := fs.String("mrg", "", "Mirage .mrg file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *sourcePath == "" || *mrgPath == "" {
		return fmt.Errorf("eval requires -source and -mrg")
	}
	sourceData, err := os.ReadFile(*sourcePath)
	if err != nil {
		return err
	}
	source, format, err := mirage.DecodeImage(bytesReader(sourceData))
	if err != nil {
		return err
	}
	mrgData, err := os.ReadFile(*mrgPath)
	if err != nil {
		return err
	}
	file, err := mirage.ParseFile(mrgData)
	if err != nil {
		return err
	}
	decoded, err := mirage.Decode(file, mirage.DefaultDecodeOptions())
	if err != nil {
		return err
	}
	mse, err := mirage.MSE(source, decoded)
	if err != nil {
		return err
	}
	psnr, err := mirage.PSNR(source, decoded)
	if err != nil {
		return err
	}
	fmt.Printf("source_format=%s image=%dx%d\n", format, source.Width, source.Height)
	fmt.Printf("mse=%.8f psnr=%.4f bpp=%.4f bytes=%d\n", mse, psnr, mirage.BitsPerPixel(file), file.Header.TotalBytes())
	return nil
}

func runInitManta(args []string) error {
	fs := flag.NewFlagSet("init-manta", flag.ExitOnError)
	out := fs.String("out", "", "output Manta .mll path")
	name := fs.String("name", "", "Manta module name")
	width := fs.Int("width", 0, "image width")
	height := fs.Int("height", 0, "image height")
	latentChannels := fs.Int("latent-channels", 0, "latent channels")
	bits := fs.Int("bits", 0, "TurboQuant bits per latent coordinate: 2, 4, or 8")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *out == "" {
		return fmt.Errorf("init-manta requires -out")
	}
	cfg := mirage.MantaConfig{
		Name:           *name,
		ImageWidth:     *width,
		ImageHeight:    *height,
		LatentChannels: *latentChannels,
		BitWidth:       *bits,
	}
	if err := mirage.WriteMantaMLL(*out, cfg); err != nil {
		return err
	}
	mod, err := mirage.MantaModule(cfg)
	if err != nil {
		return err
	}
	fp, err := mirage.MantaModelFingerprint(cfg)
	if err != nil {
		return err
	}
	fmt.Printf("wrote Manta Mirage v1 artifact -> %s\n", *out)
	fmt.Printf("module=%s entrypoints=%d steps=%d kernels=%d\n", mod.Name, len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("model_fingerprint=%s\n", hex.EncodeToString(fp[:]))
	return nil
}

func runCheckManta(args []string) error {
	fs := flag.NewFlagSet("check-manta", flag.ExitOnError)
	in := fs.String("in", "", "input Manta .mll path")
	entryFlag := fs.String("entry", "", "entrypoint to execute")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *in == "" {
		return fmt.Errorf("check-manta requires -in")
	}
	mod, err := mantaartifact.ReadFile(*in)
	if err != nil {
		return err
	}
	entryName := *entryFlag
	if entryName == "" {
		entryName = firstEntryName(mod)
	}
	entry, err := mantaEntryByName(mod, entryName)
	if err != nil {
		return err
	}
	rt := mantaruntime.New(webgpu.New())
	ctx := context.Background()
	prog, err := rt.Load(ctx, mod, mantaStubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	result, err := prog.Run(ctx, backend.Request{
		Entry:  entryName,
		Inputs: mantaStubInputs(entry),
	})
	if err != nil {
		return err
	}
	fmt.Printf("checked Manta artifact %s\n", *in)
	fmt.Printf("module=%s backend=%s entry=%s\n", mod.Name, prog.Backend(), entryName)
	fmt.Printf("outputs=%s\n", strings.Join(mantaOutputSummaries(result.Outputs), "; "))
	return nil
}

func bytesReader(data []byte) io.Reader {
	return bytes.NewReader(data)
}

func firstEntryName(mod *mantaartifact.Module) string {
	if mod != nil && len(mod.EntryPoints) > 0 {
		return mod.EntryPoints[0].Name
	}
	return ""
}

func mantaEntryByName(mod *mantaartifact.Module, name string) (mantaartifact.EntryPoint, error) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return mantaartifact.EntryPoint{}, fmt.Errorf("unknown Manta entrypoint %q", name)
}

func mantaStubLoadOptions(mod *mantaartifact.Module) []mantaruntime.LoadOption {
	opts := make([]mantaruntime.LoadOption, 0, len(mod.Params))
	for _, param := range mod.Params {
		opts = append(opts, mantaruntime.WithWeight(param.Name, mantaStubTensor(param.Type, 0.01)))
	}
	return opts
}

func mantaStubInputs(entry mantaartifact.EntryPoint) map[string]any {
	out := make(map[string]any, len(entry.Inputs))
	for _, input := range entry.Inputs {
		out[input.Name] = mantaStubTensor(input.Type, 0.1)
	}
	return out
}

func mantaStubTensor(typ mantaartifact.ValueType, offset float32) *backend.Tensor {
	if typ.Tensor == nil {
		return backend.NewTensorF32([]int{1}, []float32{offset})
	}
	shape := mantaConcreteShape(typ.Tensor.Shape)
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	switch typ.Tensor.DType {
	case "i32":
		values := make([]int32, n)
		for i := range values {
			values[i] = int32(i)
		}
		return backend.NewTensorI32(shape, values)
	case "i64":
		values := make([]int64, n)
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	case "q2":
		return backend.NewTensorQ2(shape, filledFloat32(n, offset))
	case "q4":
		return backend.NewTensorQ4(shape, filledFloat32(n, offset))
	case "q8":
		return backend.NewTensorQ8(shape, filledFloat32(n, offset))
	case "q_norm":
		return backend.NewTensorQNorm(shape, filledFloat32(n, 128))
	case "f16":
		return backend.NewTensorF16(shape, filledFloat32(n, offset))
	default:
		return backend.NewTensorF32(shape, filledFloat32(n, offset))
	}
}

func mantaConcreteShape(shape []string) []int {
	out := make([]int, len(shape))
	defaults := map[string]int{
		"B": 1,
		"C": 3,
		"D": 4,
		"E": 4,
		"H": 8,
		"N": 1,
		"T": 4,
		"V": 8,
		"W": 8,
	}
	for i, dim := range shape {
		if n, err := strconv.Atoi(dim); err == nil {
			out[i] = n
			continue
		}
		if n := defaults[dim]; n > 0 {
			out[i] = n
		} else {
			out[i] = 2
		}
	}
	return out
}

func filledFloat32(n int, offset float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = offset + float32((i%13)+1)/100
	}
	return values
}

func mantaOutputSummaries(outputs map[string]backend.Value) []string {
	keys := make([]string, 0, len(outputs))
	for key := range outputs {
		keys = append(keys, key)
	}
	sortStrings(keys)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		tensor, ok := outputs[key].Data.(*backend.Tensor)
		if !ok || tensor == nil {
			out = append(out, key+"=<non-tensor>")
			continue
		}
		out = append(out, fmt.Sprintf("%s=%s%v", key, tensor.DType, tensor.Shape))
	}
	return out
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}
