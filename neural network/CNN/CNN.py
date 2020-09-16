import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
embed_size = 300
lr, num_epochs = 0.05, 3
print_every = 20
dropout = 0.3
num_classes = 2
num_filters = 3
filter_sizes = [2,3,4]


train_on_gpu = torch.cuda.is_available()

# 读取处理好的三集及标签
train_label = np.loadtxt("train_label.txt",dtype=int)
train_set = np.loadtxt("train_set.txt",dtype=int)
valid_label = np.loadtxt("valid_label.txt",dtype=int)
valid_set = np.loadtxt("valid_set.txt",dtype=int)
test_label = np.loadtxt("test_label.txt",dtype=int)
test_set = np.loadtxt("test_set.txt",dtype=int)

# 创建TensorDataset
# torch.utils.data.TensorDataset(data_tensor, target_tensor)
# 参数：
# data_tensor (Tensor) －　包含样本数据
# target_tensor (Tensor) －　包含样本目标（标签）
train_data = TensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_label))
valid_data = TensorDataset(torch.from_numpy(valid_set), torch.from_numpy(valid_label))
test_data = TensorDataset(torch.from_numpy(test_set), torch.from_numpy(test_label))

# 创建迭代对象DataLoader，打乱数据集，设定batch size
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# 读取词典
with open("vocab.txt","r") as f:
    vocab_to_int = eval(f.read())
f.close()


class Model(nn.Module):
    def __init__(self, vocab_size,embedding_size, num_filters,filter_sizes,dropout_p, num_classes):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # 三个卷积层分别是(1, channels=256, kernal_size=(2, 300))
        #                (1, 256, (3, 300))    (1, 256, (4, 300))
        # 这三个卷积层是并行的，同时提取2-gram、3-gram、4-gram特征
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embedding_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    # 假设embed_dim=300，每个卷积层的卷积核都有256个（会将一个输入seq映射到256个channel上）
    # 三个卷积层分别为：(1, 256, (2, 300)), (1, 256, (3, 300)), (1, 256, (4, 300))
    # x(b_size, 1, seq_len, 300)进入卷积层后得到 (b, 256, seq_len-1, 1), (b, 256, seq_len-2, 1), (b, 256, seq_len-3, 1)
    # 卷积之后经过一个relu，然后把最后一个维度上的1去掉(squeeze)，得到x(b, 256, seq_len-1), 接着进入池化层
    # 一个池化层输出一个(b, 256),三个池化层输出三个(b, 256), 然后在forward里面把三个结果concat起来
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # max_pool1d表示一维池化，一维的意思是，输入x的维度除了b_size和channel，只有一维，即x(b_size, channel, d1)，故池化层只需要定义一个宽度表示kernel_size
        # max_pool2d表示二维池化，x(b_size, channel, d1, d2), 所以max_pool2d定义的kernel_size是二维的
        # max_pool1d((b, 256, seq_len-1), kernel_size = seq_len-1) -> (b, 256, 1)
        # squeeze(2) 之后得到 (b, 256)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    """
    nn中的成员比如nn.Conv2d，都是类，可以提取待学习的参数。当我们在定义网络层的时候，层内如果有需要学习的参数，那么我们就要用nn组件；
    nn.functional里的成员都是函数，只是完成一些功能，比如池化，整流线性函数，不保存参数，所以如果某一层只是单纯完成一些简单的功能，没有
    待学习的参数，那么就用nn.funcional里的组件
    """

    # 后续数据预处理时候，x被处理成是一个tuple,其形状是: (data, length).  其中data(b_size, seq_len),  length(batch_size)
    # x[0]:(b_size, seq_len)
    def forward(self, x):
        out = self.embedding(x.long())  # x[0]:(b_size, seq_len, embed_dim)    x[1]是一维的tensor,表示batch_size个元素的长度
        out = out.unsqueeze(1)  # (b_size, 1, seq_len, embed_dim)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # (b, channel * 3) == (b, 256 * 3)
        out = self.dropout(out)
        out = self.fc(out)  # out(b, num_classes)
        return out

def train(train_loader, model, epochs, print_every, batch_size):

    model.cuda()

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr )

    steps = 0
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            logit = model(inputs)
            loss = F.cross_entropy(logit, labels.long())
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % print_every == 0:
                corrects = (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()
                accuracy = 100.0 * corrects/batch_size
                print(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch_size))

def eval(data_iter, model):
    corrects, avg_loss = 0, 0

    model.eval()

    for feature, target in data_iter:
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target.long())
        avg_loss += loss.item()
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += correct

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


cnn_txt = Model(len(vocab_to_int)+1, embed_size, num_filters, filter_sizes, dropout, num_classes)
train(train_loader, cnn_txt, num_epochs, print_every, batch_size)
eval(test_loader, cnn_txt)


