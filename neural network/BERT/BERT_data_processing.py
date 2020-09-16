import numpy as np
from transformers import BertTokenizer

######## 参数设置区 ########
proportion = 0.6

######## 数据预处理 ########
bert_pre_tokenizer='F:\\神经网络\\uncased_L-12_H-768_A-12' #词表

data = []
with open("train_text.txt","rb") as f:
    for line in f:
        sentence = str(line)[2:-5]
        data.append(sentence)
with open("dev_text.txt","rb") as f:
    for line in f:
        sentence = str(line)[2:-5]
        data.append(sentence)
with open("test_text.txt","rb") as f:
    for line in f:
        sentence = str(line)[2:-5]
        data.append(sentence)
sentences=['[CLS]' + sent + '[SEP]' for sent in data]
tokenizer=BertTokenizer.from_pretrained(bert_pre_tokenizer,do_lower_case=True)
tokenized_sents=[tokenizer.tokenize(sent) for sent in sentences]

#定义句子最大长度
MAX_LEN = 30

#将分割后的句子转化成数字
input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

# PADDING
padding = []
for i in input_ids:
    if len(i) >= MAX_LEN:
        i = i[:MAX_LEN-1]
        i.append(102)
        padding.append(i)
    else:
        rest = MAX_LEN - len(i)
        for num in range(rest):
            i.append(0)
        padding.append(i)
input_ids = np.array(padding,dtype="long")

# 标签矩阵
label = []
with open('train_label.txt', 'r') as f:
    for line in f:
        label.append(line)
with open('dev_label.txt', 'r') as f:
    for line in f:
        label.append(line)
with open('test_label.txt', 'r') as f:
    for line in f:
        label.append(line)
labels = np.array([int(i) for i in label])

# 插入标签
features_with_labels = np.insert(input_ids,0,values=labels,axis=1)

# 拆分训练集、验证集、测试集
train_set = features_with_labels[:1000]
valid_set = features_with_labels[1000:1200]
test_set = features_with_labels[1200:]

# 拆分标签
train_label = train_set[:,[0]]
train_set = np.delete(train_set,0,axis=1)
valid_label = valid_set[:,[0]]
valid_set = np.delete(valid_set,0,axis=1)
test_label = test_set[:,[0]]
test_set = np.delete(test_set,0,axis=1)

# 保存矩阵
np.savetxt('bert_train_label.txt', train_label, fmt="%d", delimiter=" ")
np.savetxt('bert_train_set.txt', train_set, fmt="%d", delimiter=" ")
np.savetxt('bert_valid_label.txt', valid_label, fmt="%d", delimiter=" ")
np.savetxt('bert_valid_set.txt', valid_set, fmt="%d", delimiter=" ")
np.savetxt('bert_test_label.txt', test_label, fmt="%d", delimiter=" ")
np.savetxt('bert_test_set.txt', test_set, fmt="%d", delimiter=" ")

# 打印矩阵大小
print(train_label.shape,train_set.shape)
print(valid_label.shape,valid_set.shape)
print(test_label.shape,test_set.shape)


