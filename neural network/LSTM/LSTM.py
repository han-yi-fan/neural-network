# LSTM.py  基于LSTM神经网络的情感二分类方法
# author:韩轶凡 version:8.0
# 框架：pytorch

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


# 读取数据预处理部分已经处理好的三集及标签，转存为numpy矩阵
train_label = np.loadtxt("train_label.txt",dtype=int)
train_set = np.loadtxt("train_set.txt",dtype=int)
valid_label = np.loadtxt("valid_label.txt",dtype=int)
valid_set = np.loadtxt("valid_set.txt",dtype=int)
test_label = np.loadtxt("test_label.txt",dtype=int)
test_set = np.loadtxt("test_set.txt",dtype=int)

# 读取预处理部分形成的词典文件
with open("vocab.txt","r") as f:
    vocab_to_int = eval(f.read())
f.close()

######### 参数区 #########

# 固定参数
print_every = 20                      # 每隔20次反馈结果
vocab_size = len(vocab_to_int) + 1    # 词典长度
output_size = 1                       # 输出维度
clip = 3                              # 梯度裁剪阈值
bidirectional = True                  # 双向LSTM

# 超参数
embedding_dim = 10                    # 词嵌入维度
hidden_dim = 12                       # 隐藏层维度
n_layers = 5                          # 层数
lr = 2e-3                             # 学习率
epochs = 70                           # 训练轮次
batch_size = 50                       # 样本大小

###############################

# 设置随机种子
torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)


# 创建TensorDataset
# 实例：torch.utils.data.TensorDataset(data_tensor, target_tensor)
# 参数：
# data_tensor (Tensor) －　包含样本数据
# target_tensor (Tensor) －　包含样本目标（标签）
train_data = TensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_label))
valid_data = TensorDataset(torch.from_numpy(valid_set), torch.from_numpy(valid_label))
test_data = TensorDataset(torch.from_numpy(test_set), torch.from_numpy(test_label))

# 创建迭代对象DataLoader，打乱数据集，设定batch size
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# GPU可用标记
train_on_gpu = torch.cuda.is_available()

# 继承LSTM类
class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional=True, drop_prob=0.5):

        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,dropout=drop_prob, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout()

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):

        batch_size = x.size(0)

        # 词嵌入与输出
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        if bidirectional:
          lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        else:
          lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # 丢弃层与全连接层
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # 调整数据格式
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        # 返回最后一次sigmoid输出结果和隐层状态
        return sig_out, hidden

    def init_hidden(self, batch_size):
        # 创建隐藏层矩阵，用于保存参数，初始化各位为0
        weight = next(self.parameters()).data

        if self.bidirectional:
            number = 2
        else:
            number = 1

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_()
                      )

        return hidden

# 创建LSTM网络
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional)

# # 打印查看网络状态
# print(net)

# 损失函数为BCELoss，用来计算二分类的交叉熵
# 进行梯度优化
# parameters(memo=None) 返回一个包含模型所有参数的迭代器。一般用来当作optimizer的参数。
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


######### 训练阶段 #########

# 运算在GPU上完成
if ( train_on_gpu ): net.cuda()
net.train()
def train(epochs):
    for e in range(epochs):
        # 初始化隐藏层状态
        h = net.init_hidden(batch_size)
        counter = 0

        for inputs, labels in train_loader:
            counter += 1

            inputs, labels = inputs.cuda(), labels.cuda()

            # 为隐藏状态创建新变量
            h = tuple([each.data for each in h])
            # 梯度置0
            net.zero_grad()

            # 输出结果
            output, h = net(inputs, h)

            # 计算loss与反馈
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            # if counter % print_every == 0:
            #     net.train()
            #     print("Epoch: {}/{}...".format(e + 1, epochs),
            #           "Step: {}...".format(counter),
            #           "Loss: {:.6f}...".format(loss.item()),
            #           )


            # 按间隔在验证集上进行测试，查看效果
            if counter % print_every == 0:
                # 计算验证集loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                # 冻结网络参数，防止在验证集上进行训练
                net.eval()
                for inputs, labels in test_loader:

                    val_h = tuple([each.data for each in val_h])

                    inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

# 开始训练
train(epochs)

######### 验证阶段 #########
# 由于采用边训练边验证的方式，验证部分代码可省略
# net.eval()
# def valid():
#     val_h = net.init_hidden(batch_size)
#     val_losses = []
#     for inputs, labels in valid_loader:
#         val_h = tuple([each.data for each in val_h])
#
#         inputs, labels = inputs.cuda(), labels.cuda()
#
#         output, val_h = net(inputs, val_h)
#         val_loss = criterion(output.squeeze(), labels.float())
#
#         val_losses.append(val_loss.item())
#
#     print("Val Loss: {:.6f}".format(np.mean(val_losses)))
# valid()

######### 测试阶段 #########
# 保存测试结果
test_losses = []
num_correct = 0

# 初始化测试集的隐层状态矩阵
h = net.init_hidden(batch_size)
# 冻结网络参数
net.eval()
for inputs, labels in test_loader:

    h = tuple([each.data for each in h])

    inputs, labels = inputs.cuda(), labels.cuda()

    output, h = net(inputs, h)

    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # 输出可能性 拟合标签
    pred = torch.round(output.squeeze())

    # 计算正确分类数
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# 打印loss和acc
print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))



