# 导入必要的库
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score  # 导入轮廓系数指标
import seaborn as sns
from sklearn import metrics


# 高清显示图片
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', 'InlineBackend.figure_format="retina"')

# 保证可以显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 导入并清洗数据
raw_data = pd.read_excel('../cosinfo/cosinfoframe_precondit.xls')
raw_data = raw_data.drop(['Unnamed: 0'], axis=1)

# 绘制连续数值型数据的箱线图，检查异常值
num_cols = ['QuantitySum', 'SalesSum', 'ChargeBack', 'ChargeRatio', 'SalesperMin', 'QuantityperMin',
            'Customer']
fig = plt.figure(figsize=(12, 8))
i = 1
for col in num_cols:
    ax = fig.add_subplot(3, 5, i)
    sns.boxplot(data=raw_data[col], ax=ax)
    i = i + 1
    plt.title(col)

plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()

# 相关性分析
print('{:*^60}'.format('Correlation analysis:'))
print(raw_data.corr().round(2).T)  # 打印原始数据相关性信息
# 相关性可视化展示


corr = raw_data.corr().round(2)
sns.heatmap(corr, cmap="Reds", annot=True)


# 标准化数据
sacle_matrix = raw_data.iloc[:, 1:9]  # 获得要转换的矩阵
model_scaler = MinMaxScaler()  # 建立MinMaxScaler模型对象
data_scaled = model_scaler.fit_transform(sacle_matrix)  # MinMaxScaler标准化处理
data_scaled = pd.DataFrame(data_scaled)
data_scaled.head()

# ## PCA降维


pca = PCA(n_components=0.9999)  # 保证降维后的数据保持90%的信息，则填0.9
features = pca.fit_transform(data_scaled)

# 降维后，每个主要成分的解释方差占比（解释PC携带的信息多少）
ratio = pca.explained_variance_ratio_
print('各主成分的解释方差占比：', ratio)

# 降维后有几个成分
print('降维后有几个成分：', len(ratio))

# 累计解释方差占比
cum_ratio = np.cumsum(ratio)
print('累计解释方差占比：', cum_ratio)

# 绘制PCA降维后各成分方差占比的直方图和累计方差占比折线图
plt.figure(figsize=(8, 6))
X = range(1, len(ratio) + 1)
Y = ratio
plt.bar(X, Y, edgecolor='black')
plt.plot(X, Y, 'r.-')
plt.plot(X, cum_ratio, 'b.-')
plt.ylabel('explained_variance_ratio')
plt.xlabel('PCA')
plt.show()


# PCA选择降维保留3个主要成分
pca = PCA(n_components=3)
features2 = pca.fit_transform(features)

# 降维后的累计各成分方差占比和（即解释PC携带的信息多少）
print(sum(pca.explained_variance_ratio_))

# ## K-Means 聚类

# ![image.png](attachment:image.png)



##肘方法看k值，簇内离差平方和
sse = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(features2)
    sse.append(km.inertia_)

plt.plot(range(1, 15), sse, marker='*')
plt.xlabel('n_clusters')
plt.ylabel('distortions')
plt.title("The Elbow Method")
plt.show()




# ![image.png](attachment:image.png)
ch_scores = []
for i in range(2, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=500,
                )
    km.fit(features2)
    ch_scores.append(metrics.calinski_harabasz_score(features2, km.labels_))
plt.plot(range(2, 11), ch_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('calinski_harabaz_score')
plt.show()

# ![image.png](attachment:image.png)




# 通过平均轮廓系数检验得到最佳KMeans聚类模型
score_list = list()  # 用来存储每个K下模型的平局轮廓系数
silhouette_int = -1  # 初始化的平均轮廓系数阀值
for n_clusters in range(2, 8):  # 遍历从2到7几个有限组
    model_kmeans = KMeans(n_clusters=n_clusters, init='k-means++')  # 建立聚类模型对象
    labels_tmp = model_kmeans.fit_predict(features2)  # 训练聚类模型
    silhouette_tmp = silhouette_score(features2, labels_tmp)  # 得到每个K下的平均轮廓系数
    if silhouette_tmp > silhouette_int:  # 如果平均轮廓系数更高
        best_k = n_clusters  # 保存K将最好的K存储下来
        silhouette_int = silhouette_tmp  # 保存平均轮廓得分
        best_kmeans = model_kmeans  # 保存模型实例对象
        cluster_labels_k = labels_tmp  # 保存聚类标签
    score_list.append([n_clusters, silhouette_tmp])  # 将每次K及其得分追加到列表
print('{:*^60}'.format('K值对应的轮廓系数:'))
print(np.array(score_list))  # 打印输出所有K下的详细得分
print('最优的K值是:{0} \n对应的轮廓系数是:{1}'.format(best_k, silhouette_int))

# In[63]:


# 绘制聚类结果2维的散点图
plt.figure(figsize=(20, 8))
plt.scatter(features2[:, 0], features2[:, 1], c=cluster_labels_k)
for ii in np.arange(50):
    plt.text(features2[ii, 0], features2[ii, 1], s=ii)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means PCA')
plt.show()




plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='3d')
ax.scatter(features2[:, 0], features2[:, 1], features2[:, 2], c=cluster_labels_k)
# 视角转换，转换后更易看出簇群
ax.view_init(30, 45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# In[66]:


from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# In[70]:


# 绘制轮廓图和3d散点图
for n_clusters in range(2, 9):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(features2) + (n_clusters + 1) * 10])
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
    y_km = km.fit_predict(features2)
    silhouette_avg = silhouette_score(features2, y_km)
    print('n_cluster=', n_clusters, 'The average silhouette_score is :', silhouette_avg)

    cluster_labels = np.unique(y_km)
    silhouette_vals = silhouette_samples(features2, y_km, metric='euclidean')
    y_ax_lower = 10
    for i in range(n_clusters):
        c_silhouette_vals = silhouette_vals[y_km == i]
        c_silhouette_vals.sort()
        cluster_i = c_silhouette_vals.shape[0]
        y_ax_upper = y_ax_lower + cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(range(y_ax_lower, y_ax_upper), 0, c_silhouette_vals, edgecolor='none', color=color)
        ax1.text(-0.05, y_ax_lower + 0.5 * cluster_i, str(i))
        y_ax_lower = y_ax_upper + 10

    ax1.set_title('The silhouette plot for the various clusters')
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')

    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

    colors = cm.nipy_spectral(y_km.astype(float) / n_clusters)
    ax2.scatter(features2[:, 0], features2[:, 1], features2[:, 2], marker='.', s=30, lw=0, alpha=0.7, c=colors,
                edgecolor='k')

    centers = km.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o', c='white', alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], c[2], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    ax2.view_init(30, 45)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
plt.show()

# In[27]:


# 将原始数据与聚类标签整合
cluster_labels = pd.DataFrame(cluster_labels_k, columns=['clusters'])  # 获得训练集下的标签信息
merge_data = pd.concat((raw_data, cluster_labels), axis=1)  # 将原始处理过的数据跟聚类标签整合
merge_data.head()

# In[28]:


# 计算每个聚类类别下的样本量和样本占比
clustering_count = pd.DataFrame(merge_data['StockID'].groupby(merge_data['clusters']).count()).T.rename(
    {'StockID': 'counts'})  # 计算每个聚类类别的样本量
clustering_ratio = (clustering_count / len(merge_data)).round(2).rename({'counts': 'percentage'})  # 计算每个聚类类别的样本量占比
print(clustering_count)
print("#" * 30)
print(clustering_ratio)

# In[29]:


a = clustering_ratio.values
clustering_ratio = pd.DataFrame(a, index=['percentage'], columns=['0', '1','2'])
clustering_ratio
b = clustering_count.values
clustering_count = pd.DataFrame(b, index=['counts'], columns=['0', '1','2'])
clustering_count

# In[30]:


# 计算各个聚类类别内部最显著特征值
cluster_features = []  # 空列表，用于存储最终合并后的所有特征信
for line in range(best_k):  # 读取每个类索引
    label_data = merge_data[merge_data['clusters'] == line]  # 获得特定类的数据
    part1_data = label_data.iloc[:, 1:9]  # 获得数值型数据特征
    part1_desc = part1_data.describe().round(3)  # 得到数值型特征的描述性统计信息
    merge_data1 = part1_desc.iloc[1, :]  # 得到数值型特征的均值

    #  part2_data = label_data.iloc[:, 7:-1]  # 获得字符串型数据特征
    #   part2_desc = part2_data.describe(include='all')  # 获得字符串型数据特征的描述性统计信息
    #  merge_data2 = part2_desc.iloc[2, :]  # 获得字符串型数据特征的最频繁值

    # merge_line = pd.concat((merge_data1, merge_data2), axis=0)  # 将数值型和字符串型典型特征沿行合并
    merge_line = merge_data1
    cluster_features.append(merge_line)  # 将每个类别下的数据特征追加到列表
#  输出完整的类别特征信息
cluster_pd = pd.DataFrame(cluster_features).T  # 将列表转化为矩阵
print(cluster_pd)

# In[31]:


cluster_pd.columns = [j + f'_{i}' if cluster_pd.columns.duplicated()[i] else j for i, j in
                      enumerate(cluster_pd.columns)]
cluster_pd = cluster_pd.rename(columns={'': 'clusters', 'mean': '0', 'mean_1': '1'})
cluster_pd

# In[32]:


print('{:*^60}'.format('每个类别主要的特征:'))
all_cluster_set = pd.concat((clustering_count, cluster_pd, clustering_ratio), axis=0)  # 将每个聚类类别的所有信息合并
all_cluster_set

# In[33]:


# 各类别数据预处理
num_sets = cluster_pd.iloc[:8, :].T.astype(np.float64)  # 获取要展示的数据
num_sets_max_min = model_scaler.fit_transform(num_sets)  # 获得标准化后的数据
# 画图
fig = plt.figure(figsize=(8, 8))  # 建立画布
ax = fig.add_subplot(111, polar=True)  # 增加子网格，注意polar参数
labels = np.array(merge_data1.index)  # 设置要展示的数据标签
cor_list = ['g', 'r','b']  # , 'y', 'b']  # 定义不同类别的颜色
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)  # 计算各个区间的角度
angles = np.concatenate((angles, [angles[0]]))  # 建立相同首尾字段以便于闭合
labels = np.concatenate((labels, [labels[0]]))
# 画雷达图
for i in range(len(num_sets)):  # 循环每个类别
    data_tmp = num_sets_max_min[i, :]  # 获得对应类数据
    data = np.concatenate((data_tmp, [data_tmp[0]]))  # 建立相同首尾字段以便于闭合
    ax.plot(angles, data, 'o-', c=cor_list[i], label="第%d类渠道" % (i))  # 画线
    ax.fill(angles, data, alpha=0.8)
# 设置图像显示格式
mpl.rcParams['font.sans-serif'] = ['KaiTi']
ax.set_thetagrids(angles * 180 / np.pi, labels)  # 设置极坐标轴
ax.set_title("各聚类类别显著特征对比")  # 设置标题放置
ax.set_rlim(-0.2, 1.2)  # 设置坐标轴尺度范围
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))  # 设置图例位置

# In[35]:


name = ['SalesSum', 'ChargeBack', 'ChargeRatio', 'QuantitySum', 'Customer', 'SalesperMin', 'QuantityperMin']
for i in name:
    fig, axes = plt.subplots(1, 1)
    plt.rcParams['axes.unicode_minus'] = False
    # 解决中文乱码
    sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'FangSong']})
    a = sns.stripplot(x=merge_data['clusters'], y=merge_data[i])
    print(a)


# In[36]:


def density_plot(data):  # 自定义作图函数
    plt.figure(figsize=(10, 8), dpi=280)
    p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
    plt.title(f'第{i}簇', fontsize='large', fontweight='bold')
    [p[i].set_ylabel('density') for i in range(1)]  # 2是簇数
    plt.legend()
    return plt


pic_output = '../pic'  # 概率密度图文件名前缀
for i in range(3):
    density_plot(raw_data[merge_data[u'clusters'] == i]).savefig(u'%s%s.svg' % (pic_output, i))

# 将删除数据增添进来
drop_data = pd.read_excel('../cosinfo/data_drop1.xls')
merge_data = pd.concat([merge_data, drop_data], axis=0).drop(['Unnamed: 0','Unnamed: 0'], axis=1).fillna('异常值')


with pd.ExcelWriter('../cosinfo/result_1.xlsx') as writer:
    merge_data.to_excel(writer, sheet_name='data')


