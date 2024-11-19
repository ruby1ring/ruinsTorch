package data

import (
	"errors"
	"math/rand"
	"ruinsTorch/core"
	"time"
)

// DataLoader 接口
type IDataLoader interface {
	Reset()                                                        // 重置迭代器（开始新一轮数据迭代）
	NextBatch() ([]*core.CPUTensor, []*core.CPUTensor, error)      // 获取下一个批次的数据
	Iterate() func() ([]*core.CPUTensor, []*core.CPUTensor, error) // 返回批次迭代器
}

// DataLoader 数据加载器
type DataLoader struct {
	Dataset    IDataset // 数据集
	BatchSize  int      // 批量大小
	Shuffle    bool     // 是否打乱数据
	currentIdx int      // 当前索引
	indices    []int    // 数据索引
}

// 创建 DataLoader
func NewDataLoader(dataset IDataset, batchSize int, shuffle bool) *DataLoader {
	indices := make([]int, dataset.Len())
	for i := 0; i < dataset.Len(); i++ {
		indices[i] = i
	}
	return &DataLoader{
		Dataset:   dataset,
		BatchSize: batchSize,
		Shuffle:   shuffle,
		indices:   indices,
	}
}

// 初始化迭代器
func (d *DataLoader) Reset() {
	d.currentIdx = 0
	if d.Shuffle {
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(d.indices), func(i, j int) {
			d.indices[i], d.indices[j] = d.indices[j], d.indices[i]
		})
	}
}

// 获取下一个批次数据
func (d *DataLoader) NextBatch() ([]*core.CPUTensor, []*core.CPUTensor, error) {
	if d.currentIdx >= len(d.indices) {
		return nil, nil, errors.New("no more batches")
	}

	// 获取当前批次索引范围
	start := d.currentIdx
	end := start + d.BatchSize
	if end > len(d.indices) {
		end = len(d.indices)
	}

	// 获取数据
	batchInputs := []*core.CPUTensor{}
	batchTargets := []*core.CPUTensor{}
	for _, idx := range d.indices[start:end] {
		input, target, err := d.Dataset.GetItem(idx)
		if err != nil {
			return nil, nil, err
		}
		batchInputs = append(batchInputs, input)
		batchTargets = append(batchTargets, target)
	}

	// 更新索引
	d.currentIdx = end
	return batchInputs, batchTargets, nil
}

// 数据加载器的迭代器
func (d *DataLoader) Iterate() func() ([]*core.CPUTensor, []*core.CPUTensor, error) {
	d.Reset()
	return func() ([]*core.CPUTensor, []*core.CPUTensor, error) {
		return d.NextBatch()
	}
}
