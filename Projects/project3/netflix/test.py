import numpy as np
import em
import common

#X = np.loadtxt("test_incomplete.txt")
#X_gold = np.loadtxt("test_complete.txt")
X = np.loadtxt("netflix_complete.txt")


K = [1,12]
n, d = X.shape
seeds = [0,1,2,3,4]
LL = []
# TODO: Your code here
for s in seeds:
    mixture, post = common.init(X,K=12,seed=s)
    mixture, post , Log_likelihood = em.run(X, mixture=mixture, post=post)
    X_pred = em.fill_matrix(X, mixture)
    acc = common.rmse(X, X_pred)

    print(f"The accuracy is {acc}, seed={s} ")

    
print("\n")

