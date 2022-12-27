from scipy.stats import pearsonr
from correlation.maxcorr import *
import matplotlib.pyplot as plt

datax = list(range(-5, 6))
X = datax
datay = [x**2 for x in datax]  # quadratic data
Y = [y**2 for y in range(6)]
data = [[x, y] for x, y in zip(datax, datay)]
pearson = pearsonr(datax, datay)
maxcor = maxCorr(data, X, Y)

plt.scatter(datax,datay)
plt.title("Pearson: " + str(round(pearson[0],3)) + " | Maximal: " +str(maxcor))
plt.show()
