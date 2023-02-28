import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection as ms

cancer_data = datasets.load_breast_cancer()

y = cancer_data.target
X = cancer_data.data
X = preprocessing.scale(X)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.xlabel("tumor redius")
plt.ylabel("tumer texture")
plt.grid(True)
plt.show()


alpha = np.arange(1e-15,1,0.005)

val_scores = np.zeros((len(alpha),1))


for i in range(len(alpha)):

    model = linear_model.SGDClassifier(loss="hinge", penalty='l2', alpha=alpha[i])
    score = ms.cross_val_score(model, X, y, cv=5)

    val_scores[i] = score.mean()




plt.plot(alpha, val_scores)
plt.xlim(0,1)
plt.xlabel("alpha")
plt.ylabel("mean corss-validation accuracy")
plt.grid(True)
plt.show()


# determine the optimal alpha for high validation accuracy

ind = np.argmax(val_scores)
alpha_star = alpha[ind]
print("alpha star is {}".format(alpha_star))

print("ones looks like {}".format(np.ones(11)))
print("arange from0 to 1.1 with 0.1 step {}".format(np.arange(0,1.1,0.1)))

plt.plot(alpha, val_scores)
plt.plot(np.ones(11)*alpha_star, np.arange(0,1.1,0.1), "--r")

plt.xlim(0,1)
plt.ylim(0.94,0.98)
plt.xlabel("alpha")
plt.ylabel("mean corss-validation accuracy")
plt.grid(True)
plt.show()

# Now train the model with the optimal alpha

model_opt = linear_model.SGDClassifier(loss="hinge", penalty="l2", alpha=alpha_star)
train_model = model_opt.fit(X,y)

print("the train accuracy is {}".format(train_model.score(X,y)))

# plot the result

print(train_model.coef_)
slopes = train_model.coef_[0,1]/-train_model.coef_[0,0]

x1 = np.arange(-10,10,0.5)
Y = slopes*x1
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.plot(x1, Y, "--k")
plt.xlim(-4,4)
plt.ylim(-6,6)

plt.grid(True)
plt.show()


