import matplotlib.pyplot as plt
import pandas as pd
import time
# 解决绘图区中文乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 记时
time_start = time.time()

CosScore = pd.read_csv('Score.csv', index_col=0)
train = pd.read_csv('Traindata.csv', index_col=0)
test = pd.read_csv('Testdata.csv', index_col=0)

# 建立列表保存准确率，召回率与F值
Precision_list = list()
Recall_list = list()
F_list = list()

# 进行迭代次数推荐商品
for j in range(2, 17):  # 从1到15迭代计算推荐商品个数
    recommend_frame = pd.DataFrame(index=CosScore.index, columns=['Stock{}'.format(i for i in range(1, j))]).fillna(0)
    for i in list(CosScore.index):
        Stock_score = CosScore.loc[i]
        Stock_score = Stock_score.sort_values(ascending=False)
        for k in range(1, j):
            recommend_frame.loc[i, 'Stock{}'.format(k)] = Stock_score.index[k - 1]
    recommend_frame = recommend_frame.drop(recommend_frame.columns[0], axis=1)


    # recommend_frame.to_csv('推荐商品.csv')

    # 计算推荐系统的准确率(使用推荐为正确的商品个数/推荐商品总数)
    Customer_buy = 0
    Recommand_True = 0
    for i in list(recommend_frame.index):
        Train_set = set(recommend_frame.loc[i])
        Test_set = set()
        for k in list(train.columns):
            if train.loc[i, k] != 0:
                Test_set.add(k)
        Customer_buy += len(list(Test_set))

        Result = list(Test_set.intersection(Train_set))

        Recommand_True += len(Result)

    Recommand_All = len(recommend_frame.index) * (j - 1)
    Recommand_Precison = Recommand_True / Recommand_All
    print('Precison:{}'.format(Recommand_Precison))

    # 计算推荐系统的召回率(使用推荐为正确的商品个数/顾客购买商品总数)
    Recommand_Recall = Recommand_True / Customer_buy
    print('Recall:{}'.format(Recommand_Recall))

    # 计算F值
    F_score = 40*Recommand_Precison*Recommand_Recall/(Recommand_Precison+20*Recommand_Recall)
    print('F:{}'.format(F_score))

    # 将上述结果保存
    Precision_list.append(Recommand_Precison)
    Recall_list.append(Recommand_Recall)
    F_list.append(F_score)


    # 完成迭代
    print('第{}次迭代完成'.format(j - 1))



print(Precision_list)
print(Recall_list)
print(F_list)

# 绘制三种测量指标随推荐商品个数变化情况
x = list(range(1, len(Precision_list)+1))
plt.title('精确率，召回率，F值变化图')
plt.plot(x, Precision_list, label='精确率')
plt.plot(x, Recall_list, label='召回率')
plt.plot(x, F_list, label='F值')
plt.xlabel('推荐商品个数')
plt.legend()
plt.savefig('精确率，召回率，F值变化图')


# 当推荐商品为10的时候，F值达到最大。





# 计时
time_end = time.time()
print('time cost', time_end - time_start, 's')
