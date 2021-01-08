import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
#读取数据
movies_path = "./datasets/movies_metadata.csv"
ratings_path = "./datasets/ratings_small.csv"

movies_df = pd.read_csv(movies_path,low_memory=False)
ratings_df = pd.read_csv(ratings_path,low_memory=False)

print("movies_df size:{0}".format(movies_df.shape)) #获得dataframe的尺寸
movies_df.head()
print(movies_df.head())#获取dataframe的前五行数据

print("ratings_df size:{0}".format(ratings_df.shape))
ratings_df.head()
print(ratings_df.head())

# 数据预处理
movies_df = movies_df[['title', 'id']] #截取title和id这两列的数据
movies_df.dtypes #查看每列的数据类型
ratings_df.drop(['timestamp'], axis=1, inplace=True) #删掉timestamp列的数据
ratings_df.dtypes

#缺失值处理
#pd.to_numeric将id列的数据由字符串转为数值类型,不能转换的数据设置为NaN
print(np.where(pd.to_numeric(movies_df['id'], errors='coerce').isna())) #返回缺失值的位置，其中isna() 对于NaN返回True，否则返回False
#np.where返回满足（）内条件的数据所在的位置
movies_df.iloc[[19730, 29503, 35587]]#非法数据的位置
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce') #结果赋值给id列数据
movies_df.drop(np.where(movies_df['id'].isna())[0], inplace=True) #删除id非法的行
print("经过缺失值处理后的movies_df尺寸:{0}".format(movies_df.shape))

# 数据去重
movies_df.duplicated(['id', 'title']).sum() #返回重复项总数
movies_df.drop_duplicates(['id'], inplace=True) #数据去重
print("数据去重后movies_df的尺寸:{0}".format(movies_df.shape))

ratings_df.duplicated(['userId', 'movieId']).sum()
movies_df['id'] = movies_df['id'].astype(np.int64)##对于movies_df的id列进行类型转换
movies_df.dtypes
ratings_df.dtypes

#数据合并
#将左边的dataframe的movieId和右边的Dataframe的id进行对齐合并成新的Dataframe
ratings_df = pd.merge(ratings_df, movies_df, left_on='movieId', right_on='id')
print("数据合并后的ratings_df前五行".format(ratings_df.head()))

ratings_df.drop(['id'], axis=1, inplace=True) #去掉多余的id列
print(ratings_df.head())
print("去掉多余的id列后ratings_df size:{0}".format(ratings_df.shape))

print("有评分记录的电影个数：%d"%len(ratings_df['title'].unique())) #有评分记录的电影的个数
ratings_count = ratings_df.groupby(['title'])['rating'].count().reset_index() #统计每部电影的评分记录的总个数
print("统计每部电影的评分记录的总个数(前五行):".format(ratings_count.head()))


ratings_count = ratings_count.rename(columns={'rating':'totalRatings'}) #列的字段重命名


ratings_total = pd.merge(ratings_df,ratings_count, on='title', how='left') #添加totalRatings字段
print("添加totalRatings字段:")
print(ratings_total.head())
print(ratings_total.shape)

#数据分析以截取合适的数据
ratings_count['totalRatings'].describe()  #获得关于totalRatings字段的统计信息
ratings_count.hist()
ratings_count['totalRatings'].quantile(np.arange(.6,1,0.01)) #分位点
#由上述数据分析可知，21%的电影的评分记录个数超过20个，20即是阈值
votes_count_threshold = 20
ratings_top = ratings_total.query('totalRatings > @votes_count_threshold') #选取总评个数超过阈值的电影评分数据
print("选取总评个数超过阈值的电影评分数据")
print(ratings_top.head())
print(ratings_top.shape)
ratings_top.isna().sum() #检查有无缺失值
ratings_top.duplicated(['userId','title']).sum()  #检查是否有重复数据
ratings_top = ratings_top.drop_duplicates(['userId','title']) #只保留每个用户对每个电影的一条评分记录
ratings_top.duplicated(['userId','title']).sum()

df_for_apriori = ratings_top.pivot(index='userId',columns='title',values='rating') #调整表样式
df_for_apriori.head()

df_for_apriori = df_for_apriori.fillna(0) #缺失值填充0

def encode_units(x):  #有效评分规则，1表示有效，0表示无效
    if x <= 0:
        return 0
    if x > 0:
        return 1

df_for_apriori = df_for_apriori.applymap(encode_units) #对每个数据应用上述规则
print("数据填充完后:")
print(df_for_apriori.head())
print(df_for_apriori.shape)

#计算频繁项集和关联规则
df_for_apriori.isna().sum() #检查是否有nan值

frequent_itemsets = apriori(df_for_apriori, min_support=0.10, use_colnames=True) #生成符合条件的频繁项集
print("support降序排列的频繁项集")  #support降序排列的频繁项集
print(frequent_itemsets.sort_values('support', ascending=False))
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)  #生成关联规则，只保留lift>1的部分
print("生成关联规则，只保留lift>1的部分:")
print(rules.sort_values('lift', ascending=False))
#电影推荐

#推荐电影列表
all_antecedents = [list(x) for x in rules['antecedents'].values]
desired_indices = [i for i in range(len(all_antecedents)) if len(all_antecedents[i])==1 and all_antecedents[i][0]=='Batman Returns']
apriori_recommendations=rules.iloc[desired_indices,].sort_values(by=['lift'],ascending=False)
apriori_recommendations.head() #输出结果进行观察

apriori_recommendations_list = [list(x) for x in apriori_recommendations['consequents'].values]
print("Apriori Recommendations for movie: Batman Returns\n：")
for i in range(5):
    print("{0}: {1} with lift of {2}".format(i+1,apriori_recommendations_list[i],apriori_recommendations.iloc[i,6]))

print("\n")

#推荐单部电影
apriori_single_recommendations = apriori_recommendations.iloc[[x for x in range(len(apriori_recommendations_list)) if len(apriori_recommendations_list[x])==1],]
apriori_single_recommendations_list = [list(x) for x in apriori_single_recommendations['consequents'].values]
print("Apriori single-movie Recommendations for movie: Batman Returns\n：")
for i in range(5):
    print("{0}: {1}, with lift of {2}".format(i+1,apriori_single_recommendations_list[i][0],apriori_single_recommendations.iloc[i,6]))

#协同过滤

#只读取读取ratings_small.csv数据
ratings_path = "./datasets/ratings_small.csv"
ratings_df = pd.read_csv(ratings_path)
print(ratings_df.head(5))

#原始的movieId并非是从0或1开始的连续值，为了便于构建其user-item矩阵，我们将重新排列movie-id
movie_id = ratings_df['movieId'].drop_duplicates()
movie_id.head()
movie_id = pd.DataFrame(movie_id)
movie_id['movieid'] = range(len(movie_id))
print(len(movie_id))
print(movie_id.head())

ratings_df = pd.merge(ratings_df,movie_id,on=['movieId'],how='left')
ratings_df = ratings_df[['userId','movieid','rating','timestamp']] #更新movieId-->movieid

# 用户物品统计
n_users = ratings_df.userId.nunique()
n_items = ratings_df.movieid.nunique()
print(n_users)
print(n_items)

# 拆分数据集
# 按照训练集70%，测试集30%的比例对数据进行拆分
train_data,test_data =train_test_split(ratings_df,test_size=0.3)

# 训练集 用户-物品 矩阵
user_item_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    #print(line)
    user_item_matrix[line[1]-1,line[2]] = line[3]

# 构建用户相似矩阵 - 采用余弦距离
from sklearn.metrics.pairwise import pairwise_distances
# 相似度计算定义为余弦距离
user_similarity_m = pairwise_distances(user_item_matrix,metric='cosine')# 每个用户数据为一行，此处不需要再进行转置

user_similarity_m[0:5,0:5].round(2)
# 只分析上三角，得到等分位数
user_similarity_m_triu = np.triu(user_similarity_m,k=1) # 取得上三角数据
user_sim_nonzero = np.round(user_similarity_m_triu[user_similarity_m_triu.nonzero()],3)
np.percentile(user_sim_nonzero,np.arange(0,101,10))

#训练集预测
mean_user_rating = user_item_matrix.mean(axis=1)
rating_diff = (user_item_matrix - mean_user_rating[:,np.newaxis]) # np.newaxis作用：为mean_user_rating增加一个维度，实现加减操作

user_precdiction = mean_user_rating[:,np.newaxis] + user_similarity_m.dot(rating_diff) / np.array([np.abs(user_similarity_m).sum(axis=1)]).T
# 处以np.array([np.abs(item_similarity_m).sum(axis=1)]是为了可以使评分在1~5之间，使1~5的标准化

# 只取数据集中有评分的数据集进行评估

prediction_flatten = user_precdiction[user_item_matrix.nonzero()]
user_item_matrix_flatten = user_item_matrix[user_item_matrix.nonzero()]
error_train = sqrt(mean_squared_error(prediction_flatten,user_item_matrix_flatten)) # 均方根误差计算
print('训练集预测均方根误差：',error_train)

test_data_matrix = np.zeros((n_users,n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1]=line[3]

# 预测矩阵
rating_diff = (test_data_matrix - mean_user_rating[:,np.newaxis]) # np.newaxis作用：为mean_user_rating增加一个维度，实现加减操作
user_precdiction = mean_user_rating[:,np.newaxis] + user_similarity_m.dot(rating_diff) / np.array([np.abs(user_similarity_m).sum(axis=1)]).T

# 只取数据集中有评分的数据集进行评估
prediction_flatten = user_precdiction[user_item_matrix.nonzero()]
user_item_matrix_flatten = user_item_matrix[user_item_matrix.nonzero()]
error_test = sqrt(mean_squared_error(prediction_flatten,user_item_matrix_flatten)) # 均方根误差计算
print('测试集预测均方根误差：',error_test)