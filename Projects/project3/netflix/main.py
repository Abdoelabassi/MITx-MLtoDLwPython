import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = [1,2,3,4]
seeds = [0,1,2,3,4]
Costs = []

for i in K:
    for s in seeds:
        mixture, post = common.init(X, K=i, seed=s)

        common.plot(X, mixture=mixture, post=post, title=f"KMeans  with K={i}, and seed={s} ")

        mixture1, post, cost = kmeans.run(X, mixture=mixture, post=post)
        Costs.append(cost)

        print(f"The cost with K={i}, and seed {s}, Cost =  {cost} ")
    

print(f"Final Costs = {Costs} ")
