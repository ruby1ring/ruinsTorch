package data

import (
	"errors"
	"ruinsTorch/core"
)

// Dataset 接口
type IDataset interface {
	Len() int                                                    // 数据集大小
	GetItem(index int) (*core.CPUTensor, *core.CPUTensor, error) // 获取第 index 个样本及其标签
}

// CPUTensorDataset 简单的数据集实现
type CPUTensorDataset struct {
	Inputs  []*core.CPUTensor // 输入数据
	Targets []*core.CPUTensor // 目标数据
}

// 创建 CPUTensorDataset
func NewCPUTensorDataset(inputs []*core.CPUTensor, targets []*core.CPUTensor) *CPUTensorDataset {
	if len(inputs) != len(targets) {
		panic("inputs and targets must have the same length")
	}
	return &CPUTensorDataset{
		Inputs:  inputs,
		Targets: targets,
	}
}

// 数据集大小
func (d *CPUTensorDataset) Len() int {
	return len(d.Inputs)
}

// 获取第 index 个样本及其标签
func (d *CPUTensorDataset) GetItem(index int) (*core.CPUTensor, *core.CPUTensor, error) {
	if index < 0 || index >= len(d.Inputs) {
		return nil, nil, errors.New("index out of range")
	}
	return d.Inputs[index], d.Targets[index], nil
}
