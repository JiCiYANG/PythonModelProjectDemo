# 在神经网络中常见的用法
# 将卷积层输出展平为全连接层输入
conv_output = torch.randn(32, 64, 8, 8)  # 批次32，通道64，8×8特征图
flattened = conv_output.view(32, -1)     # 变为 [32, 64*8*8] = [32, 4096]
