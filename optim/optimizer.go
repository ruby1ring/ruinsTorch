package optim

import "ruinsTorch/core"

type Optimizer struct {
	Parameters   []*core.CPUTensor
	LearningRate float64
}

func NewOptimizer(params []*core.CPUTensor, lr float64) *Optimizer {
	return &Optimizer{Parameters: params, LearningRate: lr}
}

func (opt *Optimizer) Step() {
	for _, param := range opt.Parameters {
		for i := range param.Data {
			param.Data[i] -= opt.LearningRate * param.Grad[i] // 梯度下降
		}
	}
}
