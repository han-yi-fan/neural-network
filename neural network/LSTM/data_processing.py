# 样本数据集的清洗、文档向量化、三集的拆分和保存
import numpy as np
from string import punctuation
from collections import Counter

# 读取样本、标签文件
with open('train_text.txt', 'r') as f:
    reviews = f.read()
f.close()
with open('dev_text.txt', 'r') as f:
    reviews += f.read()
f.close()
with open('test_text.txt', 'r') as f:
    reviews += f.read()
f.close()
with open('train_label.txt', 'r') as f:
    labels = f.read()
f.close()
with open('dev_label.txt', 'r') as f:
    labels += f.read()
f.close()
with open('test_label.txt', 'r') as f:
    labels += f.read()
f.close()

# 去除标点符号
all_text = ''.join([c for c in reviews if c not in punctuation])

# 分词
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# 创建所有单词列表
words = all_text.split()
# 去停用词处理
stop_wors = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
# 创建词典
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1) if word not in stop_wors}

# 保存词典文件
with open("vocab.txt","w") as f:
    f.write(str(vocab_to_int))
f.close()

# 按词典创建文档向量矩阵
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()if word not in stop_wors])
del(reviews_ints[-1])

# # 直方图统计文档长度分布
# from matplotlib import pyplot as plt
# l = [len(i) for i in reviews_ints]
# plt.hist(l)
# plt.xlabel("X")
# plt.show()

# 类别标签矩阵
labels_split = labels.split('\n')
del(labels_split[-1])
labels = np.array([int(i) for i in labels_split])

# 对文本进行padding操作
seq_len = 10
features = []
for i in reviews_ints:
    if len(i) > seq_len:
        features.append(i[:seq_len])
    else:
        rest = seq_len - len(i)
        features.append( i+[0 for i in range(rest)]  )
features = np.array(features)


# 插入标签
features_with_labels = np.insert(features,0,values=labels,axis=1)

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
np.savetxt('train_label.txt', train_label, fmt="%d", delimiter=" ")
np.savetxt('train_set.txt', train_set, fmt="%d", delimiter=" ")
np.savetxt('valid_label.txt', valid_label, fmt="%d", delimiter=" ")
np.savetxt('valid_set.txt', valid_set, fmt="%d", delimiter=" ")
np.savetxt('test_label.txt', test_label, fmt="%d", delimiter=" ")
np.savetxt('test_set.txt', test_set, fmt="%d", delimiter=" ")

# 打印矩阵大小
print(train_label.shape,train_set.shape)
print(valid_label.shape,valid_set.shape)
print(test_label.shape,test_set.shape)





