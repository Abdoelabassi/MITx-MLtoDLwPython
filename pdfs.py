import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean = 1
variance = 2

x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x, mean, np.sqrt(variance))

plt.plot(x, pdf)
plt.show()

prob = norm.cdf(0.2, mean, np.sqrt(variance)) - norm.cdf(0.5, mean, np.sqrt(variance))
print("Probability of 0.5 < X < 0.2:", prob)
