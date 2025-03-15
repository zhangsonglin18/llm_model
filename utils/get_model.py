import torch
import torch.nn as nn
from torchviz import make_dot

# 定义一个简单的嵌入模型
class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# 初始化模型参数
vocab_size = 1000
embedding_dim = 128
model = SimpleEmbeddingModel(vocab_size, embedding_dim)

# 生成一个随机输入
input_tensor = torch.randint(0, vocab_size, (10,))

# 前向传播
output = model(input_tensor)

# 创建可视化图
dot = make_dot(output, params=dict(model.named_parameters()))

# 保存可视化图为 PDF 文件
dot.render('embedding_model_structure', format='pdf', cleanup=True)

print("网络结构已保存为 embedding_model_structure.pdf")