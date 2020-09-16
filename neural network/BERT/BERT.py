# BERT.py 基于BERT模型的情感二分类方法
# author：韩轶凡   version：5.0
# 框架：pytorch

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig,BertForSequenceClassification
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from tqdm import trange

# 设置随机种子
torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)


# 读取处理好的三集及标签并保存
train_label = np.loadtxt("bert_train_label.txt",dtype="long")
train_set = np.loadtxt("bert_train_set.txt",dtype="long")
valid_label = np.loadtxt("bert_valid_label.txt",dtype="long")
valid_set = np.loadtxt("bert_valid_set.txt",dtype="long")
test_label = np.loadtxt("bert_test_label.txt",dtype="long")
test_set = np.loadtxt("bert_test_set.txt",dtype="long")

train_set = torch.tensor(train_set).long()
train_label = torch.tensor(train_label).long()
valid_set = torch.tensor(valid_set).long()
valid_label = torch.tensor(valid_label).long()
test_set = torch.tensor(test_set).long()
test_label = torch.tensor(test_label).long()


# 建立mask机制
train_masks = []
for seq in train_set:
    seq_mask = [float(i>0) for i in seq]
    train_masks.append(seq_mask)
train_masks = np.array(train_masks)
train_masks = torch.tensor(train_masks).long()
valid_masks = []
for seq in valid_set:
    seq_mask = [float(i>0) for i in seq]
    valid_masks.append(seq_mask)
valid_masks = np.array(valid_masks)
valid_masks = torch.tensor(valid_masks).long()
test_masks = []
for seq in test_set:
    seq_mask = [float(i>0) for i in seq]
    test_masks.append(seq_mask)
test_masks = np.array(test_masks)
test_masks = torch.tensor(test_masks).long()

# 指定参数，生成dataloader
batch_size = 4

train_data = TensorDataset(train_set, train_masks, train_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(valid_set, valid_masks, valid_label)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

modelConfig = BertConfig.from_pretrained('F:\\神经网络\\uncased_L-12_H-768_A-12\\bert_config.json')
model = BertForSequenceClassification.from_pretrained('F:\\神经网络\\uncased_L-12_H-768_A-12', config=modelConfig)

# 优化参数初始化
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# 优化器及学习率
optimizer = torch.optim.Adam(optimizer_grouped_parameters,lr=2e-5,)

# 计算准确率的函数
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 训练开始
torch.cuda.empty_cache()
model.to(torch.device("cuda"))
train_loss_set = []
epochs = 1
for _ in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(torch.device("cuda")) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        #取第一个位置，BertForSequenceClassification第一个位置是Loss，第二个位置是[CLS]的logits
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
        train_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    #模型评估
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(torch.device("cuda")) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cuda').data.cpu().numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))




