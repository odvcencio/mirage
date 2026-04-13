package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

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
	case "train-manta-smoke":
		err = runTrainMantaSmoke(os.Args[2:])
	case "train-manta-kodak":
		err = runTrainMantaKodak(os.Args[2:])
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
	fmt.Fprintln(w, "  mirage train-manta-smoke -in image.png [-in image2.png] [-steps 24] [-crop 16]")
	fmt.Fprintln(w, "  mirage train-manta-kodak -dir kodak [-max-images 5] [-steps 200] [-crop 256] [-lambdas 0.001,0.01,0.1]")
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
	hyperChannels := fs.Int("hyper-channels", 0, "hyperprior channels")
	bits := fs.Int("bits", 0, "TurboQuant bits per latent coordinate: 2, 4, or 8")
	lambda := fs.Float64("lambda", 0, "rate-distortion lambda")
	factorizationFlag := fs.String("factorization", "categorical", "coordinate entropy factorization: categorical or bit-plane")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *out == "" {
		return fmt.Errorf("init-manta requires -out")
	}
	factorization, err := parseFactorization(*factorizationFlag)
	if err != nil {
		return err
	}
	cfg := mirage.MantaConfig{
		Name:           *name,
		ImageWidth:     *width,
		ImageHeight:    *height,
		LatentChannels: *latentChannels,
		HyperChannels:  *hyperChannels,
		BitWidth:       *bits,
		Factorization:  factorization,
		Lambda:         *lambda,
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

func runTrainMantaSmoke(args []string) error {
	fs := flag.NewFlagSet("train-manta-smoke", flag.ExitOnError)
	var inputs stringListFlag
	fs.Var(&inputs, "in", "input PNG, JPEG, or PPM; repeat for multiple images")
	steps := fs.Int("steps", 24, "reference SGD steps")
	lr := fs.Float64("lr", 0.02, "reference SGD learning rate")
	clip := fs.Float64("clip", 0.5, "gradient clipping threshold")
	weightDecay := fs.Float64("weight-decay", 0, "SGD weight decay")
	crop := fs.Int("crop", 16, "center crop size")
	width := fs.Int("width", 0, "center crop width; defaults to -crop")
	height := fs.Int("height", 0, "center crop height; defaults to -crop")
	latentChannels := fs.Int("latent-channels", 4, "latent channels")
	hyperChannels := fs.Int("hyper-channels", 0, "hyperprior channels; defaults to latent channels")
	bits := fs.Int("bits", 2, "TurboQuant bits per latent coordinate: 2, 4, or 8")
	lambda := fs.Float64("lambda", 0.001, "rate-distortion lambda")
	factorizationFlag := fs.String("factorization", "categorical", "coordinate entropy factorization: categorical or bit-plane")
	modelSeed := fs.Int64("model-seed", 0, "Manta graph seed")
	weightSeed := fs.Int64("weight-seed", 7, "deterministic weight initialization seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if len(inputs) == 0 {
		return fmt.Errorf("train-manta-smoke requires at least one -in image")
	}
	factorization, err := parseFactorization(*factorizationFlag)
	if err != nil {
		return err
	}
	images := make([]mirage.RGBImage, 0, len(inputs))
	formats := make([]string, 0, len(inputs))
	for _, path := range inputs {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		img, format, decodeErr := mirage.DecodeImage(f)
		closeErr := f.Close()
		if decodeErr != nil {
			return decodeErr
		}
		if closeErr != nil {
			return closeErr
		}
		images = append(images, img)
		formats = append(formats, format)
	}
	cfg := mirage.MantaConfig{
		ImageWidth:     *width,
		ImageHeight:    *height,
		LatentChannels: *latentChannels,
		HyperChannels:  *hyperChannels,
		BitWidth:       *bits,
		Seed:           *modelSeed,
		Factorization:  factorization,
		Lambda:         *lambda,
	}
	result, err := mirage.TrainMantaReferenceImages(images, mirage.MantaReferenceTrainOptions{
		Config:       cfg,
		Steps:        *steps,
		LearningRate: float32(*lr),
		GradientClip: float32(*clip),
		WeightDecay:  float32(*weightDecay),
		CropSize:     *crop,
		WeightSeed:   *weightSeed,
	})
	if err != nil {
		return err
	}
	fmt.Printf("trained Manta Mirage v1 reference smoke\n")
	fmt.Printf("images=%d formats=%s crop=%dx%d bits=%d latent_channels=%d hyper_channels=%d factorization=%s lambda=%.6g steps=%d\n",
		result.Images,
		strings.Join(formats, ","),
		result.ImageWidth,
		result.ImageHeight,
		result.BitWidth,
		result.LatentChannels,
		result.HyperChannels,
		result.Factorization,
		result.Lambda,
		result.Steps,
	)
	fmt.Printf("loss=%.6f->%.6f delta=%.6f\n", result.InitialLoss, result.FinalLoss, result.InitialLoss-result.FinalLoss)
	fmt.Printf("mse=%.6f->%.6f rate=%.6f->%.6f\n", result.InitialMSE, result.FinalMSE, result.InitialRate, result.FinalRate)
	return nil
}

func runTrainMantaKodak(args []string) error {
	fs := flag.NewFlagSet("train-manta-kodak", flag.ExitOnError)
	var inputs stringListFlag
	var dirs stringListFlag
	fs.Var(&inputs, "in", "input PNG, JPEG, or PPM; repeat or comma-separate")
	fs.Var(&dirs, "dir", "directory of PNG, JPEG, or PPM images; repeat or comma-separate")
	maxImages := fs.Int("max-images", 5, "maximum decoded images to train on; <=0 means all")
	steps := fs.Int("steps", 200, "reference SGD steps per lambda")
	crop := fs.Int("crop", 256, "center crop size")
	width := fs.Int("width", 0, "center crop width; defaults to -crop")
	height := fs.Int("height", 0, "center crop height; defaults to -crop")
	latentChannels := fs.Int("latent-channels", 4, "latent channels")
	hyperChannels := fs.Int("hyper-channels", 0, "hyperprior channels; defaults to latent channels")
	bits := fs.Int("bits", 2, "TurboQuant bits per latent coordinate: 2, 4, or 8")
	lambdaList := fs.String("lambdas", "0.001,0.01,0.1", "comma-separated lambda sweep")
	factorizationFlag := fs.String("factorization", "categorical", "coordinate entropy factorization: categorical or bit-plane")
	lr := fs.Float64("lr", 0.02, "reference SGD learning rate")
	clip := fs.Float64("clip", 0.5, "gradient clipping threshold")
	weightDecay := fs.Float64("weight-decay", 0, "SGD weight decay")
	modelSeed := fs.Int64("model-seed", 0, "Manta graph seed")
	weightSeed := fs.Int64("weight-seed", 7, "deterministic weight initialization seed")
	outDir := fs.String("out-dir", "", "optional directory for .mll modules, .weights.mll checkpoints, and summary.json")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON summary")
	if err := fs.Parse(args); err != nil {
		return err
	}
	paths, err := collectTrainingImagePaths(inputs, dirs)
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("train-manta-kodak requires at least one -in image or -dir")
	}
	if *maxImages > 0 && len(paths) > *maxImages {
		paths = paths[:*maxImages]
	}
	images, formats, err := decodeTrainingImages(paths)
	if err != nil {
		return err
	}
	lambdas, err := parseFloat64List(*lambdaList)
	if err != nil {
		return err
	}
	factorization, err := parseFactorization(*factorizationFlag)
	if err != nil {
		return err
	}
	if *outDir != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			return err
		}
	}
	summary := trainMantaKodakSummary{
		Images:  append([]string(nil), paths...),
		Formats: append([]string(nil), formats...),
		Runs:    make([]trainMantaKodakRun, 0, len(lambdas)),
	}
	for _, lambda := range lambdas {
		cfg := mirage.MantaConfig{
			ImageWidth:     *width,
			ImageHeight:    *height,
			LatentChannels: *latentChannels,
			HyperChannels:  *hyperChannels,
			BitWidth:       *bits,
			Seed:           *modelSeed,
			Factorization:  factorization,
			Lambda:         lambda,
		}
		checkpointPath := ""
		modulePath := ""
		if *outDir != "" {
			label := lambdaPathLabel(lambda)
			modulePath = filepath.Join(*outDir, "mirage_v1_lambda_"+label+".mll")
			checkpointPath = filepath.Join(*outDir, "mirage_v1_lambda_"+label+".weights.mll")
			if err := mirage.WriteMantaMLL(modulePath, cfg); err != nil {
				return err
			}
		}
		start := time.Now()
		result, err := mirage.TrainMantaReferenceImages(images, mirage.MantaReferenceTrainOptions{
			Config:         cfg,
			Steps:          *steps,
			LearningRate:   float32(*lr),
			GradientClip:   float32(*clip),
			WeightDecay:    float32(*weightDecay),
			CropSize:       *crop,
			WeightSeed:     *weightSeed,
			CheckpointPath: checkpointPath,
		})
		if err != nil {
			return err
		}
		run := trainMantaKodakRun{
			Lambda:         lambda,
			Images:         result.Images,
			Steps:          result.Steps,
			CropWidth:      result.ImageWidth,
			CropHeight:     result.ImageHeight,
			LatentChannels: result.LatentChannels,
			HyperChannels:  result.HyperChannels,
			BitWidth:       result.BitWidth,
			Factorization:  result.Factorization.String(),
			InitialLoss:    result.InitialLoss,
			FinalLoss:      result.FinalLoss,
			DeltaLoss:      result.InitialLoss - result.FinalLoss,
			InitialMSE:     result.InitialMSE,
			FinalMSE:       result.FinalMSE,
			InitialRate:    result.InitialRate,
			FinalRate:      result.FinalRate,
			Duration:       time.Since(start).String(),
			ModulePath:     modulePath,
			CheckpointPath: result.CheckpointPath,
		}
		if len(result.GradientNorms) > 0 {
			run.FirstGradientNorms = trainMantaGradientNormsFromResult(result.GradientNorms[0])
			run.LastGradientNorms = trainMantaGradientNormsFromResult(result.GradientNorms[len(result.GradientNorms)-1])
		}
		summary.Runs = append(summary.Runs, run)
		if !*jsonOut {
			fmt.Printf("lambda=%.6g images=%d crop=%dx%d steps=%d loss=%.6f->%.6f delta=%.6f mse=%.6f->%.6f rate=%.6f->%.6f grad_analysis=%.6f->%.6f grad_total=%.6f->%.6f duration=%s\n",
				run.Lambda,
				run.Images,
				run.CropWidth,
				run.CropHeight,
				run.Steps,
				run.InitialLoss,
				run.FinalLoss,
				run.DeltaLoss,
				run.InitialMSE,
				run.FinalMSE,
				run.InitialRate,
				run.FinalRate,
				run.FirstGradientNorms.Analysis,
				run.LastGradientNorms.Analysis,
				run.FirstGradientNorms.Total,
				run.LastGradientNorms.Total,
				run.Duration,
			)
		}
	}
	if *outDir != "" {
		data, err := json.MarshalIndent(summary, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*outDir, "summary.json"), append(data, '\n'), 0o644); err != nil {
			return err
		}
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(summary)
	}
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

func collectTrainingImagePaths(inputs, dirs []string) ([]string, error) {
	seen := map[string]bool{}
	var paths []string
	add := func(path string) {
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		paths = append(paths, path)
	}
	for _, path := range inputs {
		add(path)
	}
	for _, dir := range dirs {
		err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				return nil
			}
			if isTrainingImagePath(path) {
				add(path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sortStrings(paths)
	return paths, nil
}

func isTrainingImagePath(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg", ".ppm":
		return true
	default:
		return false
	}
}

func decodeTrainingImages(paths []string) ([]mirage.RGBImage, []string, error) {
	images := make([]mirage.RGBImage, 0, len(paths))
	formats := make([]string, 0, len(paths))
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return nil, nil, err
		}
		img, format, decodeErr := mirage.DecodeImage(f)
		closeErr := f.Close()
		if decodeErr != nil {
			return nil, nil, fmt.Errorf("%s: %w", path, decodeErr)
		}
		if closeErr != nil {
			return nil, nil, closeErr
		}
		images = append(images, img)
		formats = append(formats, format)
	}
	return images, formats, nil
}

func parseFloat64List(value string) ([]float64, error) {
	var out []float64
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float %q", part)
		}
		out = append(out, parsed)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one value is required")
	}
	return out, nil
}

func lambdaPathLabel(lambda float64) string {
	label := strconv.FormatFloat(lambda, 'g', -1, 64)
	label = strings.ReplaceAll(label, ".", "p")
	label = strings.ReplaceAll(label, "-", "m")
	label = strings.ReplaceAll(label, "+", "")
	return label
}

func trainMantaGradientNormsFromResult(in mirage.MantaReferenceGradientNorms) trainMantaGradientNorms {
	return trainMantaGradientNorms{
		Total:          in.Total,
		Analysis:       in.Analysis,
		HyperAnalysis:  in.HyperAnalysis,
		HyperSynthesis: in.HyperSynthesis,
		Synthesis:      in.Synthesis,
		Prior:          in.Prior,
		Other:          in.Other,
	}
}

type stringListFlag []string

func (f *stringListFlag) String() string {
	return strings.Join(*f, ",")
}

func (f *stringListFlag) Set(value string) error {
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			*f = append(*f, part)
		}
	}
	return nil
}

type trainMantaKodakSummary struct {
	Images  []string             `json:"images"`
	Formats []string             `json:"formats"`
	Runs    []trainMantaKodakRun `json:"runs"`
}

type trainMantaKodakRun struct {
	Lambda             float64                 `json:"lambda"`
	Images             int                     `json:"images"`
	Steps              int                     `json:"steps"`
	CropWidth          int                     `json:"crop_width"`
	CropHeight         int                     `json:"crop_height"`
	LatentChannels     int                     `json:"latent_channels"`
	HyperChannels      int                     `json:"hyper_channels"`
	BitWidth           int                     `json:"bit_width"`
	Factorization      string                  `json:"factorization"`
	InitialLoss        float32                 `json:"initial_loss"`
	FinalLoss          float32                 `json:"final_loss"`
	DeltaLoss          float32                 `json:"delta_loss"`
	InitialMSE         float32                 `json:"initial_mse"`
	FinalMSE           float32                 `json:"final_mse"`
	InitialRate        float32                 `json:"initial_rate"`
	FinalRate          float32                 `json:"final_rate"`
	FirstGradientNorms trainMantaGradientNorms `json:"first_gradient_norms"`
	LastGradientNorms  trainMantaGradientNorms `json:"last_gradient_norms"`
	Duration           string                  `json:"duration"`
	ModulePath         string                  `json:"module_path,omitempty"`
	CheckpointPath     string                  `json:"checkpoint_path,omitempty"`
}

type trainMantaGradientNorms struct {
	Total          float32 `json:"total"`
	Analysis       float32 `json:"analysis"`
	HyperAnalysis  float32 `json:"hyper_analysis"`
	HyperSynthesis float32 `json:"hyper_synthesis"`
	Synthesis      float32 `json:"synthesis"`
	Prior          float32 `json:"prior"`
	Other          float32 `json:"other"`
}
