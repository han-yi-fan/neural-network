import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
embed_size = 300
lr, num_epochs = 0.001, 3
print_every = 20
dropout = 0.5
num_classes = 1


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

# CNN
class CSC(nn.Module):
    def __init__(self,vocab_size,embedding_size, batch_size, dropout_p, num_classes ):
        super(CSC, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.batch_size = batch_size

        # conv2d
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(3, embedding_size),
                               stride=1) # output: [batch_size, output_channels, max_len - 3 + 1, 1]
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(3, 256),
                               stride=1) # output: [batch_size, output_channels, , 1]

        # linear
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)

        # log soft max
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,inputs):

        embedded = self.embedding(inputs.long()) # [batch_size, len]
        embedded = self.dropout(embedded)

        # conv
        conv1_output = self.conv1(embedded.unsqueeze(1))
        conv1_output = F.relu(conv1_output)
        conv1_output = conv1_output.squeeze(3).transpose(1, 2) # [batch_size, len - stride + 1, output_channels]

        conv2_output = self.conv2(conv1_output.unsqueeze(1))
        conv2_output = F.relu(conv2_output)

        # max pool [batch_size, output_channels, 1, 1]
        max_pool_output = F.max_pool2d(conv2_output, kernel_size=(conv2_output.shape[2], 1))

        # [batch_size, output_channels] out_channels
        fc1_input = max_pool_output.squeeze() #[batch_size, output_channels]

        fc1_output = self.fc1(fc1_input)
        fc2_output = self.fc2(fc1_output)

        # softmax
        output = self.softmax(fc2_output)

        return output

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
        loss = F.cross_entropy(logit, target.long(), size_average=False)
        avg_loss += loss.item()
        pred = torch.round(logit)
        correct_tensor = pred.eq(target.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        corrects += np.sum(correct)

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


cnn_txt = CSC(len(vocab_to_int)+1, embed_size, batch_size, dropout, num_classes)
train(train_loader, cnn_txt, num_epochs, print_every, batch_size)
eval(test_loader, cnn_txt)
