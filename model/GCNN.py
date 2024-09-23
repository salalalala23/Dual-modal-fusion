import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    简单的图卷积层
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

class CNNtoGCN(nn.Module):
    def __init__(self, in_channels, gcn_input_features, gcn_hidden_features, gcn_output_features):
        super(CNNtoGCN, self).__init__()
        # CNN 部分
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, gcn_input_features, kernel_size=5)

        # GCN 部分
        self.gc1 = GraphConvolution(gcn_input_features, gcn_hidden_features)
        self.gc2 = GraphConvolution(gcn_hidden_features, gcn_output_features)

        # 全连接层，用于产生最终输出
        self.fc = nn.Linear(gcn_output_features, 1)

    def forward(self, x, adj):
        # CNN 部分
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # 将 CNN 输出转换为 GCN 输入
        batch_size, gcn_feats, h, w = x.shape
        x = x.view(batch_size, gcn_feats, -1).permute(0, 2, 1).contiguous()
        x = x.view(batch_size * h * w, gcn_feats)

        # GCN 部分
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)

        # 将 GCN 输出转换回批量格式
        x = x.view(batch_size, h * w, -1).mean(dim=1)

        # 全连接层
        x = self.fc(x)
        return x.squeeze()

# 模型参数
in_channels = 4 # 图像通道数
gcn_input_features = 32
gcn_hidden_features = 16
gcn_output_features = 10

# 创建模型实例
model = CNNtoGCN(in_channels, gcn_input_features, gcn_hidden_features, gcn_output_features)

# 假设 'images' 是 BCHW 格式的图像数据，'adj' 是邻接矩阵
images = torch.randn([20, 4, 64, 64])
adj = torch.zeros([3380, 3380])

# 使用模型
output = model(images, adj)
