package nn

import "ruinsTorch/core"

type Module interface {
	Forward(input *core.CPUTensor) (*core.CPUTensor, error) // 前向传播
	Parameters() []*core.CPUTensor                          // 获取模型中的所有参数
	ZeroGrad()                                              // 清空所有参数的梯度
}

type BaseModule struct {
	params []*core.CPUTensor // 存储模型参数
}

// 添加参数
func (m *BaseModule) AddParameter(t *core.CPUTensor) {
	m.params = append(m.params, t)
}

// 获取参数
func (m *BaseModule) Parameters() []*core.CPUTensor {
	return m.params
}

// 清空梯度
func (m *BaseModule) ZeroGrad() {
	for _, p := range m.params {
		p.ZeroGrad()
	}
}
