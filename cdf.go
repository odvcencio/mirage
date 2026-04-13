package mirage

import (
	"fmt"
	"math"
	"sort"
)

// CDFFromProbabilities converts non-negative probabilities into an integer CDF
// suitable for the deterministic 32-bit arithmetic coder.
func CDFFromProbabilities(probabilities []float32, total uint32) (CDF, error) {
	if len(probabilities) == 0 {
		return CDF{}, fmt.Errorf("mirage: empty probability vector")
	}
	if total == 0 {
		total = maxCDFTotal
	}
	if total > maxCDFTotal {
		return CDF{}, fmt.Errorf("mirage: CDF total %d exceeds %d", total, maxCDFTotal)
	}
	sum := 0.0
	positive := 0
	for i, p := range probabilities {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) || p < 0 {
			return CDF{}, fmt.Errorf("mirage: invalid probability at %d", i)
		}
		if p > 0 {
			sum += float64(p)
			positive++
		}
	}
	if positive == 0 {
		return CDF{}, fmt.Errorf("mirage: probability vector has no mass")
	}
	if positive > int(total) {
		return CDF{}, fmt.Errorf("mirage: %d positive probabilities exceed CDF total %d", positive, total)
	}

	type residual struct {
		index int
		frac  float64
	}
	counts := make([]uint32, len(probabilities))
	residuals := make([]residual, 0, len(probabilities))
	assigned := uint32(0)
	for i, p := range probabilities {
		if p == 0 {
			continue
		}
		scaled := float64(p) / sum * float64(total)
		count := uint32(math.Floor(scaled))
		if count == 0 {
			count = 1
		}
		counts[i] = count
		assigned += count
		residuals = append(residuals, residual{index: i, frac: scaled - math.Floor(scaled)})
	}
	if assigned < total {
		sort.SliceStable(residuals, func(i, j int) bool {
			if residuals[i].frac == residuals[j].frac {
				return residuals[i].index < residuals[j].index
			}
			return residuals[i].frac > residuals[j].frac
		})
		for assigned < total {
			for _, r := range residuals {
				if assigned == total {
					break
				}
				counts[r.index]++
				assigned++
			}
		}
	}
	if assigned > total {
		sort.SliceStable(residuals, func(i, j int) bool {
			if residuals[i].frac == residuals[j].frac {
				return residuals[i].index > residuals[j].index
			}
			return residuals[i].frac < residuals[j].frac
		})
		for assigned > total {
			changed := false
			for _, r := range residuals {
				if assigned == total {
					break
				}
				if counts[r.index] <= 1 {
					continue
				}
				counts[r.index]--
				assigned--
				changed = true
			}
			if !changed {
				return CDF{}, fmt.Errorf("mirage: could not normalize CDF to total %d", total)
			}
		}
	}
	return NewCDF(counts)
}

// CDFFromLogits converts logits into an integer CDF using a stable softmax.
func CDFFromLogits(logits []float32, total uint32) (CDF, error) {
	if len(logits) == 0 {
		return CDF{}, fmt.Errorf("mirage: empty logits vector")
	}
	maxLogit := math.Inf(-1)
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return CDF{}, fmt.Errorf("mirage: invalid logit at %d", i)
		}
		if float64(v) > maxLogit {
			maxLogit = float64(v)
		}
	}
	probabilities := make([]float32, len(logits))
	for i, v := range logits {
		probabilities[i] = float32(math.Exp(float64(v) - maxLogit))
	}
	return CDFFromProbabilities(probabilities, total)
}

// CDFsFromProbabilities converts a flattened tensor of probability vectors into
// one CDF per vector. The final dimension is alphabet.
func CDFsFromProbabilities(probabilities []float32, alphabet int, total uint32) ([]CDF, error) {
	if alphabet <= 0 {
		return nil, fmt.Errorf("mirage: alphabet must be positive")
	}
	if len(probabilities)%alphabet != 0 {
		return nil, fmt.Errorf("mirage: probability tensor length %d is not divisible by alphabet %d", len(probabilities), alphabet)
	}
	models := make([]CDF, len(probabilities)/alphabet)
	for i := range models {
		start := i * alphabet
		model, err := CDFFromProbabilities(probabilities[start:start+alphabet], total)
		if err != nil {
			return nil, fmt.Errorf("mirage: probability vector %d: %w", i, err)
		}
		models[i] = model
	}
	return models, nil
}

// CDFsFromLogits converts a flattened tensor of logits into one CDF per vector.
// The final dimension is alphabet, matching the v1 pi_logits tensor layouts.
func CDFsFromLogits(logits []float32, alphabet int, total uint32) ([]CDF, error) {
	if alphabet <= 0 {
		return nil, fmt.Errorf("mirage: alphabet must be positive")
	}
	if len(logits)%alphabet != 0 {
		return nil, fmt.Errorf("mirage: logits tensor length %d is not divisible by alphabet %d", len(logits), alphabet)
	}
	models := make([]CDF, len(logits)/alphabet)
	for i := range models {
		start := i * alphabet
		model, err := CDFFromLogits(logits[start:start+alphabet], total)
		if err != nil {
			return nil, fmt.Errorf("mirage: logits vector %d: %w", i, err)
		}
		models[i] = model
	}
	return models, nil
}

// NormCDFsFromLogNormalParams converts flattened (mu, sigma) log-normal
// parameters into one q_norm CDF per spatial position.
func NormCDFsFromLogNormalParams(params []float32, total uint32) ([]CDF, error) {
	return NormCDFsFromLogNormalParamsWithQuantizer(params, total, DefaultNorms)
}

// NormCDFsFromLogNormalParamsWithQuantizer is NormCDFsFromLogNormalParams with
// an explicit q_norm quantizer. params is flattened as [mu0, sigma0, ...].
func NormCDFsFromLogNormalParamsWithQuantizer(params []float32, total uint32, q NormQuantizer) ([]CDF, error) {
	if err := q.validate(); err != nil {
		return nil, err
	}
	if len(params)%2 != 0 {
		return nil, fmt.Errorf("mirage: norm params length %d is not pairs of (mu, sigma)", len(params))
	}
	models := make([]CDF, len(params)/2)
	for i := range models {
		mu := float64(params[i*2])
		sigma := float64(params[i*2+1])
		if math.IsNaN(mu) || math.IsInf(mu, 0) || math.IsNaN(sigma) || math.IsInf(sigma, 0) || sigma <= 0 {
			return nil, fmt.Errorf("mirage: invalid norm params at %d", i)
		}
		probs := make([]float32, 256)
		for sym := range probs {
			lower, upper := q.normSymbolLogBounds(sym)
			mass := normalCDF((upper-mu)/sigma) - normalCDF((lower-mu)/sigma)
			if mass < 0 && mass > -1e-12 {
				mass = 0
			}
			probs[sym] = float32(mass)
		}
		model, err := CDFFromProbabilities(probs, total)
		if err != nil {
			return nil, fmt.Errorf("mirage: norm params %d: %w", i, err)
		}
		models[i] = model
	}
	return models, nil
}

func (q NormQuantizer) normSymbolLogBounds(sym int) (float64, float64) {
	span := q.LogMax - q.LogMin
	switch sym {
	case 0:
		return math.Inf(-1), q.LogMin + (0.5/255)*span
	case 255:
		return q.LogMin + (254.5/255)*span, math.Inf(1)
	default:
		lower := (float64(sym) - 0.5) / 255
		upper := (float64(sym) + 0.5) / 255
		return q.LogMin + lower*span, q.LogMin + upper*span
	}
}

func normalCDF(x float64) float64 {
	switch {
	case math.IsInf(x, -1):
		return 0
	case math.IsInf(x, 1):
		return 1
	default:
		return 0.5 * (1 + math.Erf(x/math.Sqrt2))
	}
}
