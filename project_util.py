import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering



###################################################################
# sort eigen value and vector in ascendent order
#####################################################################
def argEig(w, v):
    
    a = w.argsort()
    new_w = w[a]
    
    v1 = v.transpose()
    v2 = v1[a]
    new_v = v2.transpose()
    return new_w, new_v

#################################################################
# calculate feature score
###############################################################
def calcPhi(f, w, v, k=-2): # the lower this score the better the feature
    w_ = w[1:(k+1)]
    v_ = v[:,1:(k+1)]
    a = f.dot(v_)
    a2 = a.dot(a)
    a3 = a**2
    return a3.dot(w_) / a2


#################################################################
# generate the T indicator A.K.A the labels
##################################################################
def T(high, low, close, threshold=0.025, timeperiod=14):
    HLC = (high + low + close) / 3
    
    x = []
    n = len(close)
    for i in range(0, n):
        si = 0.0
        for j in range(1, timeperiod+1):
            if (i+j < n):
                ti = (HLC[i+j] - close[i]) / close[i]
                if (ti < -threshold or ti > threshold):
                    si += ti

        x.append(si)

    x = np.array(x)

    return x

################################################################
# get eig values and vectors of the laplacian in ascert order
################################################################
def solvLap(x, n_clusters=8, random_state=123, affinity='rbf', gamma=0.0001):
    spc = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity=affinity, gamma=gamma)

    #spc.fit_predict(x)
    spc.fit(x)
    
    #print(spc.affinity_matrix_)
    #print(spc.labels_)
    
    A = spc.affinity_matrix_
    D = np.diag(A.sum(axis=1))
    
    L = D - A
    
    w, v = np.linalg.eig(L)
    
    
    w_, v_ = argEig(w, v)

    return w_, v_
