from project_util import *
import pandas as pd
iris = load_iris()

x, y = iris.data, iris.target

spc = SpectralClustering(n_clusters=3, random_state=123, affinity='rbf')

#spc.fit_predict(x)
spc.fit(x)

#print(spc.affinity_matrix_)
#print(spc.labels_)

A = spc.affinity_matrix_
D = np.diag(A.sum(axis=1))

L = D - A

w, v = np.linalg.eig(L)


w_, v_ = argEig(w, v)

#print(w_)
#print(v_)
#print(x[:,0])
#print(calcPhi(x[:,0],w_, v_))

def phi(x):
    return calcPhi(x, w_, v_)
score = x.apply(phi, axis=1)

print(score)
