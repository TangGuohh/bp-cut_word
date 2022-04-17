# 深度学习模型框架
from gensim.models import word2vec
# 分词常用工具
import jieba
# 深度学习框架 包含许多库函数以及基础模型
import torch
from torch.nn.functional import cosine_similarity

# 什么是校园卡 校园卡是什么
# 昆明理工大学在哪里 昆明理工大学的地址
s1 = '什么是校园卡'
s2 = '校园卡是什么'
s3 = '昆明理工大学在哪里'
s4 = '昆明理工大学地址'
s5 = '年长的校长吃西瓜'
s6 = '年轻的组长吃香瓜'
# 句子集合
sens = [s1, s2, s3, s4, s5, s6]
# 加载停用词
with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]
# 保存所有的词汇
words_all = []
# 文本预处理
for s in sens:
    # 切分词汇
    jieba.add_word("昆明理工大学")
    words = jieba.lcut(s)
    temp = []
    for wd in words:
        if wd not in stopwords:
            temp.append(wd)
    words_all.append(temp)
# 调用词向量模型
model = word2vec.Word2Vec(words_all, min_count=1)
# 、、print(model.wv.similarity('昆明理工大学', '地址'))
# 、、print(model.wv.similarity('保时捷', '法拉利'))
vec = model.wv['昆明理工大学']
vecs = []
print(words_all)
for s in words_all:
    vs = []
    for wd in s:
        v = model.wv[wd]
        vs.append(v)
    vecs.append(vs)
all_s = []
for v in vecs:
    s1 = []
    for i in v:
        if len(s1) == 0:
            s1 = i
        else:
            s1 = s1 + i
    all_s.append(s1 / len(v))
print(all_s)
sim1 = cosine_similarity(torch.from_numpy(all_s[0]).view(1, -1),
                        torch.from_numpy(all_s[1]).view(1, -1))
sim2 = cosine_similarity(torch.from_numpy(all_s[2]).view(1, -1),
                        torch.from_numpy(all_s[3]).view(1, -1))
sim3 = cosine_similarity(torch.from_numpy(all_s[4]).view(1, -1),
                        torch.from_numpy(all_s[5]).view(1, -1))
# sim3 = cosine_similarity(all_s[0], all_s[1])
print(sim1.item())
print(sim2.item())
print(sim3.item())
pass


