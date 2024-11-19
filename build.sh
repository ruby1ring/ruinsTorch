
# 创建核心功能模块目录及文件
mkdir core
touch core/tensor.go
touch core/autograd.go
touch core/utils.go

# 创建神经网络模块目录及文件
mkdir nn
touch nn/module.go
touch nn/linear.go
touch nn/activation.go
touch nn/loss.go

# 创建优化器模块目录及文件
mkdir optim
touch optim/sgd.go
touch optim/adam.go
touch optim/optimizer.go

# 创建训练工具模块目录及文件
mkdir train
touch train/engine.go
touch train/dataloader.go

# 创建项目配置文件和入口文件
touch go.mod
touch main.go

# 确保目录结构正确
tree
