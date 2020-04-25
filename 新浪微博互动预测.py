import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
import xgboost as xgb

data_train=pd.DataFrame(pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\新浪微博互动预测\weibo_train_data.txt',sep='\t'))
data_test=pd.DataFrame(pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\新浪微博互动预测\weibo_predict_data.txt',sep='\t'))
data_train.columns=['uid','mid','time','forward_count','comment_count','like_count','content']
data_train['type']='train'
data_test.columns=['uid','mid','time','content']
data_test['type']='test'
'''print('转发:',data_train['forward_count'].describe().astype(int))
print('评论:',data_train['comment_count'].describe().astype(int))
print('点赞:',data_train['like_count'].describe().astype(int))'''
#print(data_train.info())
#print(data_test.info())
data_all=pd.concat([data_train,data_test],axis=0,sort=False)
data_all['time']=pd.to_datetime(data_all['time'])
data_all['month']=data_all.time.dt.month
data_all['hour']=data_all.time.dt.hour
data_all['weekday']=data_all.time.dt.weekday
'''data_all['hour'].value_counts().sort_index().plot(kind='bar')
plt.show()'''
#temp=data_all.loc[data_all['month']==2|3|4|5,'content']
def hour_cut(x):
    if 0<=x<=7: #凌晨
        return 0
    elif  7<x<=12: #上午
        return 1
    elif  12<x<=17: # 下午
        return 2
    elif  17<x<=19: # 傍晚
        return 3
    elif  19<x<24: # 晚上
        return 4
data_all['hour_cut']=data_all['hour'].map(hour_cut)
data_all=data_all.drop(['hour'],axis=1)
###### 特征工程 用户特征
# 评论特征
uid_and_commentCount=data_train.groupby('uid')['comment_count'].count()
uid_and_commentMean=data_train.groupby('uid')['comment_count'].mean()
uid_and_commentMax=data_train.groupby('uid')['comment_count'].max()

# 点赞特征
uid_and_likeCount=data_train.groupby('uid')['like_count'].count()
uid_and_likeMean=data_train.groupby('uid')['like_count'].mean()
uid_and_likeMax=data_train.groupby('uid')['like_count'].max()

# 转发特征
uid_and_forwardCount=data_train.groupby('uid')['forward_count'].count()
uid_and_forwardMean=data_train.groupby('uid')['forward_count'].mean()
uid_and_forwardMax=data_train.groupby('uid')['forward_count'].max()

# 博文特征
uid_and_contentCount=data_train.groupby('uid')['content'].count()

'''content=dict(data_all['content'])
contentjieba=jieba.lcut(content)
counts={}
for word in contentjieba:
    if len(word)==1:
        continue
    elif word==',':
        continue
    else:
        counts[word] = counts.get(word, 0) + 1
items=list(counts.items())
items.sort(key=lambda x: x[1], reverse=True)
for i in range(15):
    word, count = items[i]
    print("{0:<5}{1:>5}".format(word, count))
print(content)'''

# 数据合并
data_all['uid_and_commentCount']=data_all.loc[:,'uid'].map(uid_and_commentCount).fillna(0)
data_all['uid_and_commentMean']=data_all.loc[:,'uid'].map(uid_and_commentMean).fillna(0)
data_all['uid_and_commentMax']=data_all.loc[:,'uid'].map(uid_and_commentMax).fillna(0)
data_all['uid_and_likeCount']=data_all.loc[:,'uid'].map(uid_and_likeCount).fillna(0)
data_all['uid_and_likeMean']=data_all.loc[:,'uid'].map(uid_and_likeMean).fillna(0)
data_all[' uid_and_likeMax']=data_all.loc[:,'uid'].map( uid_and_likeMax).fillna(0)
data_all['uid_and_forwardCount']=data_all.loc[:,'uid'].map(uid_and_forwardCount).fillna(0)
data_all['uid_and_forwardMean']=data_all.loc[:,'uid'].map(uid_and_forwardMean).fillna(0)
data_all['uid_and_forwardMax']=data_all.loc[:,'uid'].map(uid_and_forwardMax).fillna(0)
data_all['uid_and_contentCount']=data_all.loc[:,'uid'].map(uid_and_contentCount).fillna(0)

data_all['http']=0
data_all['hongbao']=0
data_all['fengxiang']=0
data_all['dache']=0
data_all['cn']=0
data_all['weibo']=0
data_all['topic']=0
data_all['ai']=0
data_all['zhuangfa']=0
data_all['daijinjuan']=0
data_all['nianfen']=0
temp=data_all.loc[0:100,'content'].index
for index in temp:
    seg_list = jieba.cut(data_all.loc[index,'content'].to_string())
    for j in seg_list:
        if j=='http':
            data_all.loc[index,'http']=1
        elif j=='红包':
            data_all.loc[index,'hongbao']=1
        elif j=='分享':
            data_all.loc[index,'fengxiang']=1
        elif j=='打车':
            data_all.loc[index,'dache']=1
        elif j=='cn':
            data_all.loc[index,'cn']=1
        elif j=='微博':
            data_all.loc[index,'weibo']=1
        elif j=='##':
            data_all.loc[index,'topic']=1
        elif j=='@':
            data_all.loc[index,'ai']=1
        elif j=='[':
            data_all.loc[index,'zhuangfa']=1
        elif j=='代金券':
            data_all.loc[index,'daijinjuan']=1
        elif j=='2015':
            data_all.loc[index,'nianfen']=1

data_all=data_all.drop(['uid','mid','time','content','type'],axis=1)
train1=data_all.loc[data_all['month']==2 | 3 | 4,:]
test1=data_all.loc[data_all['month']==5,:]
'''train2=data_all.loc[data_all['month']== 3 | 4 | 5,:]
test2=data_all.loc[data_all['month']==6,:]
train3=data_all.loc[data_all['month']==4 | 5 | 6,:]
test3=data_all.loc[data_all['month']==7,:]'''
y_train=train1.loc[:,'forward_count','comment_count','like_count']
X_train=train1.drop(['forward_count','comment_count','like_count'],axis=1)

y_test=test1.loc[:,'forward_count','comment_count','like_count']
X_test=test1.drop(['forward_count','comment_count','like_count'],axis=1)

model_xgb= xgb.XGBRegressor(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2);
model_xgb.fit(X_train,y_train)
xgb_pred=model_xgb.predict(X_test)
print('mse:',mean_squared_error(y_test,xgb_pred))
print('正确率:',accuracy_score(y_test,xgb_pred))
