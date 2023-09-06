import numpy as np
import pandas as pd



'''导入数据'''


predict_target = pd.read_excel('D:\研究生课程\研究生案例大赛\实验\ecommercedata-预测.xlsx',index_col=0)
data = pd.read_excel('D:\研究生课程\研究生案例大赛\实验\cosinfo\CustomerInfo_3.xlsx', index_col=0)


'''寻找测试集内没有在之前购买过商品的客户'''
predict_target_set = set()
data_set = set()
for i in list(predict_target.index):
    predict_target_set.add(i)
for j in list(data.index):
    data_set.add(j)

predict_difference = predict_target_set.difference(data_set)
print(predict_difference)

predict_target_drop = list(predict_difference)
predict_target = predict_target.drop(list(predict_difference), axis=0)

'''寻找测试集内没有在之前购买过商品的客户'''
predict_target_set = set()
data_set = set()
for i in list(predict_target.index):
    predict_target_set.add(i)
for j in list(data.index):
    data_set.add(j)

predict_difference = predict_target_set.difference(data_set)
print(predict_difference)



'''寻找最近的K个邻居顾客'''
'''定义距离函数'''
# 定义欧氏距离
def euSimilar(a,b):
    dist = np.linalg.norm(a - b)
    return 1 / (dist+0.0001)

# 定义余弦相似度
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


# 寻找最相似的K个客户
SimilarCos = dict() # 创建一个字典存储最相似的客户与得分
for i in list(predict_target.index):
    distance_list = [0 for h in range(5)]
    SimilarCos_list = [0 for k in range(5)]
    for j in list(data.index.drop(i)):
        x = np.array(data.loc[i])
        y = np.array(data.loc[j])
        distance = cos_sim(x, y) # 计算余弦相似度
        if distance > min(distance_list):
            a = distance_list.index(min(distance_list))
            distance_list[a] = distance
            SimilarCos_list[a] = j
        else:
            continue

    Total_Simsocre = np.array(distance_list, dtype='float64').sum()  # 对相似度得分进行求和
    SimScore = [k/Total_Simsocre for k in distance_list] # 求得平均相似度得分

    SimilarCos[i] = [SimilarCos_list, SimScore]

print(SimilarCos)

# 将最相似顾客与其得分写入Excel表格中
rowname=list()
for headers in SimilarCos.keys():
    rowname.append(headers)
SimilarCos_Frame = pd.DataFrame(index=rowname, columns=['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5',
                                                        'Score1', 'Score2','Score3', 'Score4', 'Score5' ]).fillna(0)
for keys in SimilarCos.keys():
    list1 = SimilarCos[keys][0]+list(SimilarCos[keys][1])
    SimilarCos_Frame.loc[keys] = list1

print(SimilarCos_Frame)

'''执行推荐算法'''
'''将数据集进行标准化（逐列）'''
for i in list(data.columns):
    Mean = np.mean(data.loc[:, i])
    Std = np.std(data.loc[:, i])
    if Std != 0:
        data.loc[:, i] = (data.loc[:, i]-Mean)/Std
    else:
        continue
print(data)

'''使用推荐算法'''
# 算法思路，将最接近的K个客户所购买的商品数量标准分数，以余弦相似度为权重，计算加权平均数作为指数得分，最终得分最高的n类商品为推荐商品。
# 使用推荐评分算法
CosScore = pd.DataFrame(index=predict_target.index, columns=data.columns).fillna(0)
for i in list(SimilarCos_Frame.index):
    s1 = SimilarCos_Frame.loc[i, list(SimilarCos_Frame.columns)[0:5]] # 用来保存最相似客户的ID
    s2 = SimilarCos_Frame.loc[i, list(SimilarCos_Frame.columns)[5:]] # 用来保存最相似客户对应的相似度得分
    ScoreList = np.array(list(0 for i in range(len(CosScore.columns))), dtype='float64') # 用来保存相似度得分列表
    Total_s2 = 0 # 计算余弦相似度加权使用

    for j in range(len(s1)):
        if s1[j] != 0 and s2[j] != 0:
            c = np.array(data.loc[s1[j]], dtype='float64')*s2[j] # 以余弦相似度作为权重 计算指数评分
        ScoreList += c
        Total_s2 += np.array(s2[j])
    CosScore.loc[i] = ScoreList/Total_s2






# 进行推荐商品
recommend_frame = pd.DataFrame(index=CosScore.index,columns=['Stock{}'.format(i for i in range(1,11))]).fillna(0)
for i in list(CosScore.index):
    Stock_score = CosScore.loc[i]
    Stock_score = Stock_score.sort_values(ascending=False)
    for k in range(1, 11):
        recommend_frame.loc[i, 'Stock{}'.format(k)] = Stock_score.index[k-1]

print(recommend_frame)

recommend_frame.to_csv('推荐商品_Predict.csv')







