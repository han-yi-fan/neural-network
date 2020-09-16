# transformer.py 基于transformer神经网络的情感二分类方法
# author：韩轶凡    version：7.0
# 模型框架：pytorch

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics


# 位置编码函数，与句向量维度一致，相加用以表示单词在句子位置
class Positional_Encoding(nn.Module):

    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        # 参考网络资料选用sin、cos的奇偶位置编码方法
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 单词embedding与位置编码相加，这两个张量的shape一致
        out = x + nn.Parameter(self.pe, requires_grad=False)
        out = self.dropout(out)
        return out


# 多头注意力机制
class Multi_Head_Attention(nn.Module):

    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 按照head数量进行张量均分

        # 计算Q、K、V三个参数矩阵
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)  # K
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)  # V
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)   # 规范化处理

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # 调整矩阵形状为 batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        # 计算注意力
        scale = K.size(-1) ** -0.5               # Scaled操作
        context = self.attention(Q, K, V, scale) # 论文中提到的"Scaled_Dot_Product_Attention"计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape回原形状
        # 全连接+丢弃层
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x      # 残差连接,ADD
        out = self.layer_norm(out)  # 规范化
        return out


# Scaled Dot-Product 实现
class Scaled_Dot_Product_Attention(nn.Module):

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


#  位置前馈层实现
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


# 编码器实现
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


# Transformer模型
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab, config.embed)
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size,config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多层Encoder

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        out0 = self.embedding(x.long())
        out = self.postion_embedding(out0)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)   # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        out = self.fc1(out)
        return out


# 模型参数存放
class ConfigTrans(object):

    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 2             # 分类数
        self.num_epochs = 80             # 训练轮次
        self.batch_size = 50             # mini-batch大小
        self.pad_size = 15               # 每句话处理成的长度
        self.learning_rate = 5e-4        # 学习率
        self.embed = 100                 # 词嵌入维度
        self.dim_model = self.embed      # 模型维度
        self.hidden = 15                 # 隐藏层维度
        self.num_head = 10               # 多头注意力，注意需要整除
        self.num_encoder = 3             # 编码器层数
        self.class_list = ["0","1"]      # 类别标签
        self.vocab = 3500                # 词典长度


# 训练函数
def train(config, model, train_iter, dev_iter, test_iter):
    model.train()
    # 优化方法设置
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()            # 学习率衰减
        for trains, labels in train_iter:
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels.long())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data
                predic = torch.max(outputs.data, 1)[1]
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                model.train()
            total_batch += 1
    test(config, model, test_iter)


# 测试函数
def test(config, model, test_iter):
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


# 用于计算二分类的各分类正确率
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels.long())
            loss_total += loss
            labels = labels.data
            predic = torch.max(outputs.data, 1)[1]
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



# 读取处理好的三集及标签
# train_label = np.loadtxt("train_label.txt",dtype=int)
# train_set = np.loadtxt("train_set.txt",dtype=int)
# valid_label = np.loadtxt("valid_label.txt",dtype=int)
# valid_set = np.loadtxt("valid_set.txt",dtype=int)
# test_label = np.loadtxt("test_label.txt",dtype=int)
# test_set = np.loadtxt("test_set.txt",dtype=int)

# 读取词典
# with open("vocab.txt","r") as f:
#     vocab_to_int = eval(f.read())
# f.close()

# 随机种子设置
torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)

# 转为tensordataset类型存放
train_data = TensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_label))
valid_data = TensorDataset(torch.from_numpy(valid_set), torch.from_numpy(valid_label))
test_data = TensorDataset(torch.from_numpy(test_set), torch.from_numpy(test_label))

# 设置参数
config = ConfigTrans()

# 创建迭代对象DataLoader，打乱数据集，设定batch size
train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=config.batch_size)
test_loader = DataLoader(test_data, batch_size=config.batch_size)

# 初始化物理模型
net = Transformer()

# 开始训练+验证+测试
train(config, net,train_loader,valid_loader,test_loader)