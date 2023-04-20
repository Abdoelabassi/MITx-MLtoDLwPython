import numpy as np
import em
import common

#X = np.loadtxt("test_incomplete.txt")
#X_gold = np.loadtxt("test_complete.txt")
X = np.loadtxt("toy_data.txt")


K = [1,2,3,4]
n, d = X.shape
seed = [0,1,2,3,4]
LL = []
# TODO: Your code here
for k in K:
    mixture, post = common.init(X,K=k,seed=0)
    mixture, post , Log_likelihood = em.run(X, mixture=mixture, post=post)
    common.plot(X, mixture, post, title=f"EM algo with K={k}")
    print(f"Log likelihood={Log_likelihood}, k={k}, and seed={0} ")
    LL.append(Log_likelihood)

print("\n")


