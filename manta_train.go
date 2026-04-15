//go:build !js

package mirage

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mantamodels "github.com/odvcencio/manta/models"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	_ "github.com/odvcencio/manta/runtime/backends/cuda"
)

const defaultMantaReferenceTrainCropSize = 16

// MantaReferenceTrainOptions controls a tiny image-backed reference training
// run over the Manta Mirage v1 graph.
type MantaReferenceTrainOptions struct {
	Config               MantaConfig
	Steps                int
	LearningRate         float32
	LearningRateSchedule string
	FinalLearningRate    float32
	LambdaSchedule       string
	InitialLambda        float32
	LambdaDelaySteps     int
	LambdaRampSteps      int
	GradientClip         float32
	WeightDecay          float32
	MinGDNBeta           float32
	Optimizer            string
	AdamBeta1            float32
	AdamBeta2            float32
	AdamEpsilon          float32
	FreezeAnalysisSteps  int
	CropSize             int
	CropMode             string
	RandomCrops          int
	CropSeed             int64
	WeightSeed           int64
	ResumePath           string
	ResumeOptimizerPath  string
	ScheduleSteps        int
	Backend              string
	CheckpointPath       string
	CheckpointEvery      int
	CheckpointPrefix     string
}

// MantaReferenceTrainResult summarizes convergence for one reference run.
type MantaReferenceTrainResult struct {
	Images              int
	Steps               int
	ImageWidth          int
	ImageHeight         int
	LatentChannels      int
	HyperChannels       int
	BitWidth            int
	Factorization       Factorization
	Lambda              float64
	CropMode            string
	TrainingCrops       int
	RandomCrops         int
	CropSeed            int64
	ResumePath          string
	ResumeOptimizerPath string
	Backend             string
	InitialStep         int
	ScheduleSteps       int
	InitialLoss         float32
	FinalLoss           float32
	InitialMSE          float32
	FinalMSE            float32
	InitialRate         float32
	FinalRate           float32
	Losses              []float32
	MSEs                []float32
	Rates               []float32
	LearningRates       []float32
	Lambdas             []float32
	GradientNorms       []MantaReferenceGradientNorms
	Checkpoints         []MantaReferenceCheckpoint
	CheckpointPath      string
	OptimizerPath       string
	Optimizer           string
	LearningRate        float32
	LRSchedule          string
	FinalLR             float32
	LambdaSchedule      string
	InitialLambda       float32
	LambdaDelay         int
	LambdaRamp          int
	FreezeAnalysis      int
}

// MantaReferenceCheckpoint records metrics and optional artifact paths for a
// periodic reference training checkpoint.
type MantaReferenceCheckpoint struct {
	Step           int
	Loss           float32
	MSE            float32
	Rate           float32
	LearningRate   float32
	Lambda         float32
	ModulePath     string
	CheckpointPath string
	OptimizerPath  string
}

// MantaReferenceGradientNorms records raw reference autograd gradient L2 norms
// by Mirage graph region before clipping.
type MantaReferenceGradientNorms struct {
	Total          float32
	Analysis       float32
	HyperAnalysis  float32
	HyperSynthesis float32
	Synthesis      float32
	Prior          float32
	Other          float32
}

// TrainMantaReferenceImages trains the Manta Mirage v1 graph on decoded RGB
// images using the CPU reference autograd path. Images are cropped according to
// opts.CropMode before they are converted to NCHW tensors.
func TrainMantaReferenceImages(images []RGBImage, opts MantaReferenceTrainOptions) (MantaReferenceTrainResult, error) {
	if len(images) == 0 {
		return MantaReferenceTrainResult{}, fmt.Errorf("mirage: at least one training image is required")
	}
	opts = normalizeMantaReferenceTrainOptions(opts)
	if opts.ScheduleSteps < 0 {
		return MantaReferenceTrainResult{}, fmt.Errorf("mirage: schedule steps must be non-negative")
	}
	mod, err := MantaModule(opts.Config)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	tensors, err := mantaReferenceTrainTensors(images, opts)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	weights, err := mantaReferenceInitialWeights(mod, opts)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	optimizerState, resumeScheduleSteps, err := mantaReferenceInitialOptimizerState(opts)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	initialStep := 0
	if optimizerState != nil {
		initialStep = optimizerState.Step
	}
	scheduleSteps := mantaReferenceEffectiveScheduleSteps(opts, initialStep, resumeScheduleSteps)
	imageGradAccel, err := mantaReferenceImageGradAccelerator(opts.Backend)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	if imageGradAccel != nil {
		defer imageGradAccel.Close()
	}
	checkpointFunc := mantaReferenceCheckpointFunc(opts, scheduleSteps)
	history, err := mantamodels.TrainMirageV1Reference(mod, weights, tensors, mantamodels.MirageV1ReferenceTrainConfig{
		Steps:                opts.Steps,
		LearningRate:         opts.LearningRate,
		LearningRateSchedule: opts.LearningRateSchedule,
		FinalLearningRate:    opts.FinalLearningRate,
		LambdaSchedule:       opts.LambdaSchedule,
		InitialLambda:        opts.InitialLambda,
		LambdaDelaySteps:     opts.LambdaDelaySteps,
		LambdaRampSteps:      opts.LambdaRampSteps,
		GradientClip:         opts.GradientClip,
		WeightDecay:          opts.WeightDecay,
		MinGDNBeta:           opts.MinGDNBeta,
		Optimizer:            opts.Optimizer,
		AdamBeta1:            opts.AdamBeta1,
		AdamBeta2:            opts.AdamBeta2,
		AdamEpsilon:          opts.AdamEpsilon,
		FreezeAnalysisSteps:  opts.FreezeAnalysisSteps,
		InitialStep:          initialStep,
		ScheduleSteps:        scheduleSteps,
		OptimizerState:       optimizerState,
		CheckpointEvery:      opts.CheckpointEvery,
		CheckpointStateFunc:  checkpointFunc,
		ImageGradAccelerator: imageGradAccel,
	})
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	if opts.CheckpointPath != "" {
		if err := mantaruntime.NewWeightFile(weights).WriteFile(opts.CheckpointPath); err != nil {
			return MantaReferenceTrainResult{}, err
		}
		if err := writeMantaReferenceOptimizerState(mantaReferenceOptimizerPathForWeights(opts.CheckpointPath), opts, scheduleSteps, history.OptimizerState); err != nil {
			return MantaReferenceTrainResult{}, err
		}
	}
	return MantaReferenceTrainResult{
		Images:              len(images),
		Steps:               opts.Steps,
		ImageWidth:          opts.Config.ImageWidth,
		ImageHeight:         opts.Config.ImageHeight,
		LatentChannels:      opts.Config.LatentChannels,
		HyperChannels:       opts.Config.HyperChannels,
		BitWidth:            opts.Config.BitWidth,
		Factorization:       opts.Config.Factorization,
		Lambda:              opts.Config.Lambda,
		CropMode:            opts.CropMode,
		TrainingCrops:       len(tensors),
		RandomCrops:         opts.RandomCrops,
		CropSeed:            opts.CropSeed,
		ResumePath:          opts.ResumePath,
		ResumeOptimizerPath: mantaReferenceResumeOptimizerPath(opts),
		Backend:             opts.Backend,
		InitialStep:         initialStep,
		ScheduleSteps:       scheduleSteps,
		InitialLoss:         history.InitialLoss,
		FinalLoss:           history.FinalLoss,
		InitialMSE:          history.InitialMSE,
		FinalMSE:            history.FinalMSE,
		InitialRate:         history.InitialRate,
		FinalRate:           history.FinalRate,
		Losses:              append([]float32(nil), history.Losses...),
		MSEs:                append([]float32(nil), history.MSEs...),
		Rates:               append([]float32(nil), history.Rates...),
		LearningRates:       append([]float32(nil), history.LearningRates...),
		Lambdas:             append([]float32(nil), history.Lambdas...),
		GradientNorms:       convertMantaReferenceGradientNorms(history.GradientNorms),
		Checkpoints:         convertMantaReferenceCheckpoints(history.Checkpoints, opts.CheckpointPrefix, opts.Optimizer),
		CheckpointPath:      opts.CheckpointPath,
		OptimizerPath:       mantaReferenceSavedOptimizerPath(opts, opts.CheckpointPath),
		Optimizer:           opts.Optimizer,
		LearningRate:        opts.LearningRate,
		LRSchedule:          opts.LearningRateSchedule,
		FinalLR:             opts.FinalLearningRate,
		LambdaSchedule:      opts.LambdaSchedule,
		InitialLambda:       opts.InitialLambda,
		LambdaDelay:         opts.LambdaDelaySteps,
		LambdaRamp:          opts.LambdaRampSteps,
		FreezeAnalysis:      opts.FreezeAnalysisSteps,
	}, nil
}

func mantaReferenceCheckpointFunc(opts MantaReferenceTrainOptions, scheduleSteps int) func(mantamodels.MirageV1ReferenceCheckpoint, map[string]*backend.Tensor, *mantamodels.MirageV1ReferenceOptimizerState) error {
	if opts.CheckpointEvery <= 0 || opts.CheckpointPrefix == "" {
		return nil
	}
	return func(checkpoint mantamodels.MirageV1ReferenceCheckpoint, weights map[string]*backend.Tensor, optimizerState *mantamodels.MirageV1ReferenceOptimizerState) error {
		modulePath, checkpointPath, optimizerPath := mantaReferenceCheckpointPaths(opts.CheckpointPrefix, checkpoint.Step)
		if err := WriteMantaMLL(modulePath, opts.Config); err != nil {
			return err
		}
		if err := mantaruntime.NewWeightFile(weights).WriteFile(checkpointPath); err != nil {
			return err
		}
		if err := writeMantaReferenceOptimizerState(optimizerPath, opts, scheduleSteps, optimizerState); err != nil {
			return err
		}
		return nil
	}
}

func mantaReferenceCheckpointPaths(prefix string, step int) (string, string, string) {
	stem := fmt.Sprintf("%s_step_%06d", prefix, step)
	return stem + ".mll", stem + ".weights.mll", stem + ".optim.mll"
}

func convertMantaReferenceGradientNorms(in []mantamodels.MirageV1ReferenceGradientNorms) []MantaReferenceGradientNorms {
	if len(in) == 0 {
		return nil
	}
	out := make([]MantaReferenceGradientNorms, len(in))
	for i, item := range in {
		out[i] = MantaReferenceGradientNorms{
			Total:          item.Total,
			Analysis:       item.Analysis,
			HyperAnalysis:  item.HyperAnalysis,
			HyperSynthesis: item.HyperSynthesis,
			Synthesis:      item.Synthesis,
			Prior:          item.Prior,
			Other:          item.Other,
		}
	}
	return out
}

func convertMantaReferenceCheckpoints(in []mantamodels.MirageV1ReferenceCheckpoint, prefix, optimizer string) []MantaReferenceCheckpoint {
	if len(in) == 0 {
		return nil
	}
	out := make([]MantaReferenceCheckpoint, len(in))
	for i, item := range in {
		out[i] = MantaReferenceCheckpoint{
			Step:         item.Step,
			Loss:         item.Loss,
			MSE:          item.MSE,
			Rate:         item.Rate,
			LearningRate: item.LearningRate,
			Lambda:       item.Lambda,
		}
		if prefix != "" {
			out[i].ModulePath, out[i].CheckpointPath, out[i].OptimizerPath = mantaReferenceCheckpointPaths(prefix, item.Step)
			if optimizer != "adam" {
				out[i].OptimizerPath = ""
			}
		}
	}
	return out
}

func normalizeMantaReferenceTrainOptions(opts MantaReferenceTrainOptions) MantaReferenceTrainOptions {
	opts.Backend = strings.ToLower(strings.TrimSpace(opts.Backend))
	if opts.Backend == "" {
		opts.Backend = "reference"
	}
	if opts.CropSize == 0 {
		opts.CropSize = defaultMantaReferenceTrainCropSize
	}
	if opts.Config.ImageWidth == 0 {
		opts.Config.ImageWidth = opts.CropSize
	}
	if opts.Config.ImageHeight == 0 {
		opts.Config.ImageHeight = opts.CropSize
	}
	if opts.Config.ImageChannels == 0 {
		opts.Config.ImageChannels = 3
	}
	if opts.Config.LatentChannels == 0 {
		opts.Config.LatentChannels = 4
	}
	if opts.Config.HyperChannels == 0 {
		opts.Config.HyperChannels = opts.Config.LatentChannels
	}
	if opts.Config.BitWidth == 0 {
		opts.Config.BitWidth = 2
	}
	if opts.Config.Lambda == 0 && !opts.Config.LambdaSet {
		opts.Config.Lambda = 0.001
	}
	if opts.Steps <= 0 {
		opts.Steps = 16
	}
	if opts.LearningRate == 0 {
		opts.LearningRate = 0.01
	}
	opts.LearningRateSchedule = strings.ToLower(strings.TrimSpace(opts.LearningRateSchedule))
	if opts.LearningRateSchedule == "" {
		opts.LearningRateSchedule = "constant"
	}
	opts.LambdaSchedule = strings.ToLower(strings.TrimSpace(opts.LambdaSchedule))
	if opts.LambdaSchedule == "" {
		opts.LambdaSchedule = "constant"
	}
	if opts.FinalLearningRate == 0 {
		opts.FinalLearningRate = opts.LearningRate
	}
	opts.Optimizer = strings.ToLower(strings.TrimSpace(opts.Optimizer))
	if opts.Optimizer == "" {
		opts.Optimizer = "sgd"
	}
	if opts.GradientClip == 0 {
		opts.GradientClip = 1
	}
	opts.CropMode = strings.ToLower(strings.TrimSpace(opts.CropMode))
	if opts.CropMode == "" {
		opts.CropMode = "center"
	}
	if opts.RandomCrops <= 0 {
		opts.RandomCrops = 1
	}
	if opts.CropSeed == 0 {
		opts.CropSeed = DefaultSeed
	}
	if opts.WeightSeed == 0 {
		opts.WeightSeed = DefaultSeed
	}
	return opts
}

func mantaReferenceImageGradAccelerator(name string) (backend.ImageGradAccelerator, error) {
	switch name {
	case "reference", "cpu":
		return nil, nil
	case "cuda":
		accel, _, err := backend.NewPreferredImageGradAccelerator(mantaartifact.BackendCUDA)
		if err != nil {
			return nil, err
		}
		if accel == nil {
			return nil, fmt.Errorf("mirage: CUDA image-gradient backend is unavailable")
		}
		return accel, nil
	default:
		return nil, fmt.Errorf("mirage: unknown training backend %q", name)
	}
}

func mantaReferenceTrainTensors(images []RGBImage, opts MantaReferenceTrainOptions) ([]*backend.Tensor, error) {
	switch opts.CropMode {
	case "center":
		tensors := make([]*backend.Tensor, 0, len(images))
		for i, img := range images {
			tensor, err := rgbImageToMantaTrainTensor(img, opts.Config.ImageWidth, opts.Config.ImageHeight)
			if err != nil {
				return nil, fmt.Errorf("training image %d: %w", i, err)
			}
			tensors = append(tensors, tensor)
		}
		return tensors, nil
	case "random":
		rng := rand.New(rand.NewSource(opts.CropSeed))
		tensors := make([]*backend.Tensor, 0, len(images)*opts.RandomCrops)
		for i, img := range images {
			for crop := 0; crop < opts.RandomCrops; crop++ {
				tensor, err := randomRGBImageToMantaTrainTensor(img, opts.Config.ImageWidth, opts.Config.ImageHeight, rng)
				if err != nil {
					return nil, fmt.Errorf("training image %d crop %d: %w", i, crop, err)
				}
				tensors = append(tensors, tensor)
			}
		}
		return tensors, nil
	default:
		return nil, fmt.Errorf("mirage: unknown crop mode %q", opts.CropMode)
	}
}

func mantaReferenceInitialWeights(mod *mantaartifact.Module, opts MantaReferenceTrainOptions) (map[string]*backend.Tensor, error) {
	if opts.ResumePath != "" {
		weights, err := mantaruntime.ReadWeightFile(opts.ResumePath)
		if err != nil {
			return nil, err
		}
		if err := validateMantaReferenceWeights(mod, weights.Weights); err != nil {
			return nil, fmt.Errorf("resume weights: %w", err)
		}
		return weights.Weights, nil
	}
	return mantamodels.InitMirageV1ReferenceWeights(mod, opts.WeightSeed)
}

const (
	mantaReferenceAdamStepTensor          = "__mirage_adam_step"
	mantaReferenceScheduleStepsTensor     = "__mirage_schedule_steps"
	mantaReferenceLearningRateTensor      = "__mirage_learning_rate"
	mantaReferenceFinalLearningRateTensor = "__mirage_final_learning_rate"
	mantaReferenceAdamBeta1Tensor         = "__mirage_adam_beta1"
	mantaReferenceAdamBeta2Tensor         = "__mirage_adam_beta2"
	mantaReferenceAdamEpsilonTensor       = "__mirage_adam_epsilon"
	mantaReferenceAdamMPrefix             = "adam_m/"
	mantaReferenceAdamVPrefix             = "adam_v/"
)

func mantaReferenceInitialOptimizerState(opts MantaReferenceTrainOptions) (*mantamodels.MirageV1ReferenceOptimizerState, int, error) {
	if opts.Optimizer != "adam" {
		return nil, 0, nil
	}
	path := mantaReferenceResumeOptimizerPath(opts)
	if path == "" {
		return nil, 0, nil
	}
	weights, err := mantaruntime.ReadWeightFile(path)
	if err != nil {
		if opts.ResumeOptimizerPath == "" && os.IsNotExist(err) {
			return nil, 0, nil
		}
		return nil, 0, fmt.Errorf("resume optimizer state: %w", err)
	}
	step, err := scalarInt64Tensor(weights.Weights, mantaReferenceAdamStepTensor)
	if err != nil {
		return nil, 0, err
	}
	scheduleSteps, err := optionalScalarInt64Tensor(weights.Weights, mantaReferenceScheduleStepsTensor)
	if err != nil {
		return nil, 0, err
	}
	state := &mantamodels.MirageV1ReferenceOptimizerState{
		Step: int(step),
		M:    map[string]*backend.Tensor{},
		V:    map[string]*backend.Tensor{},
	}
	for name, tensor := range weights.Weights {
		switch {
		case strings.HasPrefix(name, mantaReferenceAdamMPrefix):
			state.M[strings.TrimPrefix(name, mantaReferenceAdamMPrefix)] = tensor.Clone()
		case strings.HasPrefix(name, mantaReferenceAdamVPrefix):
			state.V[strings.TrimPrefix(name, mantaReferenceAdamVPrefix)] = tensor.Clone()
		}
	}
	if len(state.M) == 0 || len(state.V) == 0 {
		return nil, 0, fmt.Errorf("resume optimizer state has no Adam moments")
	}
	return state, int(scheduleSteps), nil
}

func writeMantaReferenceOptimizerState(path string, opts MantaReferenceTrainOptions, scheduleSteps int, state *mantamodels.MirageV1ReferenceOptimizerState) error {
	if path == "" || opts.Optimizer != "adam" || state == nil {
		return nil
	}
	tensors := map[string]*backend.Tensor{
		mantaReferenceAdamStepTensor:          backend.NewTensorI64([]int{1}, []int64{int64(state.Step)}),
		mantaReferenceScheduleStepsTensor:     backend.NewTensorI64([]int{1}, []int64{int64(scheduleSteps)}),
		mantaReferenceLearningRateTensor:      backend.NewTensorF32([]int{1}, []float32{opts.LearningRate}),
		mantaReferenceFinalLearningRateTensor: backend.NewTensorF32([]int{1}, []float32{opts.FinalLearningRate}),
		mantaReferenceAdamBeta1Tensor:         backend.NewTensorF32([]int{1}, []float32{opts.AdamBeta1}),
		mantaReferenceAdamBeta2Tensor:         backend.NewTensorF32([]int{1}, []float32{opts.AdamBeta2}),
		mantaReferenceAdamEpsilonTensor:       backend.NewTensorF32([]int{1}, []float32{opts.AdamEpsilon}),
	}
	for name, tensor := range state.M {
		if tensor != nil {
			tensors[mantaReferenceAdamMPrefix+name] = tensor.Clone()
		}
	}
	for name, tensor := range state.V {
		if tensor != nil {
			tensors[mantaReferenceAdamVPrefix+name] = tensor.Clone()
		}
	}
	return mantaruntime.NewWeightFile(tensors).WriteFile(path)
}

func mantaReferenceResumeOptimizerPath(opts MantaReferenceTrainOptions) string {
	if opts.ResumeOptimizerPath != "" {
		return opts.ResumeOptimizerPath
	}
	if opts.ResumePath == "" {
		return ""
	}
	return mantaReferenceOptimizerPathForWeights(opts.ResumePath)
}

func mantaReferenceOptimizerPathForWeights(path string) string {
	if path == "" {
		return ""
	}
	if strings.HasSuffix(path, ".weights.mll") {
		return strings.TrimSuffix(path, ".weights.mll") + ".optim.mll"
	}
	if strings.HasSuffix(path, ".mll") {
		return strings.TrimSuffix(path, ".mll") + ".optim.mll"
	}
	ext := filepath.Ext(path)
	if ext == "" {
		return path + ".optim.mll"
	}
	return strings.TrimSuffix(path, ext) + ".optim.mll"
}

func mantaReferenceSavedOptimizerPath(opts MantaReferenceTrainOptions, weightPath string) string {
	if opts.Optimizer != "adam" {
		return ""
	}
	return mantaReferenceOptimizerPathForWeights(weightPath)
}

func mantaReferenceEffectiveScheduleSteps(opts MantaReferenceTrainOptions, initialStep, resumeScheduleSteps int) int {
	if opts.ScheduleSteps > 0 {
		return opts.ScheduleSteps
	}
	if resumeScheduleSteps > 0 {
		return resumeScheduleSteps
	}
	return initialStep + opts.Steps
}

func scalarInt64Tensor(tensors map[string]*backend.Tensor, name string) (int64, error) {
	value, err := optionalScalarInt64Tensor(tensors, name)
	if err != nil {
		return 0, err
	}
	if value == 0 {
		if tensor := tensors[name]; tensor == nil {
			return 0, fmt.Errorf("optimizer state missing %s", name)
		}
	}
	return value, nil
}

func optionalScalarInt64Tensor(tensors map[string]*backend.Tensor, name string) (int64, error) {
	tensor := tensors[name]
	if tensor == nil {
		return 0, nil
	}
	if len(tensor.I64) == 1 {
		return tensor.I64[0], nil
	}
	if len(tensor.I32) == 1 {
		return int64(tensor.I32[0]), nil
	}
	if len(tensor.F32) == 1 {
		return int64(tensor.F32[0]), nil
	}
	return 0, fmt.Errorf("optimizer state tensor %s is not scalar", name)
}

func rgbImageToMantaTrainTensor(img RGBImage, width, height int) (*backend.Tensor, error) {
	if err := validateMantaTrainCrop(img, width, height); err != nil {
		return nil, err
	}
	x0 := (img.Width - width) / 2
	y0 := (img.Height - height) / 2
	return rgbImageCropToMantaTrainTensor(img, width, height, x0, y0)
}

func randomRGBImageToMantaTrainTensor(img RGBImage, width, height int, rng *rand.Rand) (*backend.Tensor, error) {
	if rng == nil {
		return nil, fmt.Errorf("mirage: random crop generator is nil")
	}
	if err := validateMantaTrainCrop(img, width, height); err != nil {
		return nil, err
	}
	x0 := 0
	if maxX := img.Width - width; maxX > 0 {
		x0 = rng.Intn(maxX + 1)
	}
	y0 := 0
	if maxY := img.Height - height; maxY > 0 {
		y0 = rng.Intn(maxY + 1)
	}
	return rgbImageCropToMantaTrainTensor(img, width, height, x0, y0)
}

func rgbImageCropToMantaTrainTensor(img RGBImage, width, height, x0, y0 int) (*backend.Tensor, error) {
	if err := validateMantaTrainCrop(img, width, height); err != nil {
		return nil, err
	}
	if x0 < 0 || y0 < 0 || x0+width > img.Width || y0+height > img.Height {
		return nil, fmt.Errorf("mirage: crop origin %d,%d is outside image %dx%d for crop %dx%d", x0, y0, img.Width, img.Height, width, height)
	}
	values := make([]float32, 3*height*width)
	for c := 0; c < 3; c++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				values[(c*height+y)*width+x] = img.Pix[img.offset(c, y0+y, x0+x)]
			}
		}
	}
	return backend.NewTensorF16([]int{1, 3, height, width}, values), nil
}

func validateMantaTrainCrop(img RGBImage, width, height int) error {
	if err := img.validate(); err != nil {
		return err
	}
	if width <= 0 || height <= 0 {
		return fmt.Errorf("mirage: training crop dimensions must be positive")
	}
	if width%16 != 0 || height%16 != 0 {
		return fmt.Errorf("mirage: training crop must be divisible by 16")
	}
	if img.Width < width || img.Height < height {
		return fmt.Errorf("mirage: image %dx%d is smaller than requested crop %dx%d", img.Width, img.Height, width, height)
	}
	return nil
}

func validateMantaReferenceWeights(mod *mantaartifact.Module, weights map[string]*backend.Tensor) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if len(weights) == 0 {
		return fmt.Errorf("no weights")
	}
	for _, param := range mod.Params {
		if param.Type.Tensor == nil {
			return fmt.Errorf("param %q is not a tensor", param.Name)
		}
		weight := weights[param.Name]
		if weight == nil {
			return fmt.Errorf("missing weight %q", param.Name)
		}
		shape, err := concreteMantaParamShape(param.Type.Tensor.Shape)
		if err != nil {
			return fmt.Errorf("param %q shape: %w", param.Name, err)
		}
		if !sameIntShape(weight.Shape, shape) {
			return fmt.Errorf("weight %q shape %v does not match module shape %v", param.Name, weight.Shape, shape)
		}
		if len(weight.F32) != weight.Elements() {
			return fmt.Errorf("weight %q storage has %d values, want %d", param.Name, len(weight.F32), weight.Elements())
		}
	}
	return nil
}

func concreteMantaParamShape(shape []string) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		n, err := strconv.Atoi(dim)
		if err != nil {
			return nil, fmt.Errorf("non-concrete dim %q", dim)
		}
		if n <= 0 {
			return nil, fmt.Errorf("non-positive dim %d", n)
		}
		out[i] = n
	}
	return out, nil
}

func sameIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
