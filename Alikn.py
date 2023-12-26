import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# sklearn库
from sklearn.cluster import KMeans
import pandas as ps
from BHT_ARIMA import BHTARIMA
from BHT_ARIMA.util.utility import get_index
from BHT_ARIMA.util.utility import wavelet_noising
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift

from datetime import datetime


def predict(ts,knPred):
    knts = ts[np.where(ts[:,10]==knPred)]
    #ts=np.delete(ts,3,axis=1)
    # 421

    p = 4  # p-order
    d = 2  # d-order
    q = 1  # q-order
    taus = [11, 5]  # MDT-rank
    Rs = [5, 5]  # tucker decomposition ranks
    k = 10  # iterations
    tol = 0.001  # stop criterion
    Us_mode = 4  # orthogonality mode

    # Run program
    # result's shape: (ITEM, TIME+1) ** only one step forecasting **
    model = BHTARIMA(knts.T, p, d, q, taus, Rs, k, tol, verbose=0, Us_mode=Us_mode)
    result, _ = model.run()
    onePred = np.round(result[..., -1],4)
    ts = np.append(ts, onePred.reshape(1, 11), axis=0)

    return ts;

before = datetime.now()

ori_ts = np.load('./input/Ali_afterHandle.npy', allow_pickle=True)
pred_len = 1
ori_ts=np.delete(ori_ts,0,axis=1)
#ori_ts = ori_ts[:-3,...]
label = ori_ts[-pred_len:,... ].T  # label, take the last time step as label
ts = ori_ts[:-pred_len,...]
#a =ts[:, 16].reshape(-1,1)

#效果主要改变这个值
n_clusters = 3
# 创建模型(定义K-Means聚类)
modelKn=KMeans(n_clusters=n_clusters,max_iter=10,init='k-means++')

# 训练模型
modelKn.fit(ts[:, 9].reshape(-1,1))

#print("聚类中心(质心)：",modelKn.cluster_centers_)
print("每个样本所属的簇(类别)：",modelKn.labels_)
labels_=modelKn.labels_.reshape(-1,1)
#标准化
sc = MinMaxScaler(feature_range=(0, 1))
ori_ts = sc.fit_transform(ori_ts)
#np.insert(ori_ts,3,model.labels_,axis=1)
ts=np.c_[ts,labels_]

# zero = ts[np.where(ts[:, 10] == 0)]
# one = ts[np.where(ts[:, 10] == 1)]
# two = ts[np.where(ts[:, 10] == 2)]
# three = ts[np.where(ts[:, 10] == 3)]
# four = ts[np.where(ts[:, 10] == 4)]

# 111

p = 1  # p-order
d = 1  # d-order
q = 1  # q-order
taus = [1, 5]  # MDT-rank
Rs = [5, 5]  # tucker decomposition ranks
k = 10  # iterations
tol = 0.001  # stop criterion
Us_mode = 4  # orthogonality mode




# Run program
# result's shape: (ITEM,, TIME+1) ** only one step forecasting **
for i in range(pred_len):
    model = BHTARIMA(labels_.T, p, d, q, taus, Rs, k, tol, verbose=0, Us_mode=Us_mode)
    result, _ = model.run()
    knPred = np.round(result[..., -1])
    labels_=np.append(labels_,knPred.reshape(-1,1),axis=0)
    ts=predict(ts,knPred)

finalPred=np.delete(ts,10,axis=1)
#finalPred=sc.inverse_transform(finalPred)
finalPred=finalPred[-pred_len:,... ].T

print(finalPred[-1, ...])

print("Evaluation single index: \n{}".format(get_index(finalPred[-1, ...], label[-1, ...])))

after = datetime.now()

print(after-before)



