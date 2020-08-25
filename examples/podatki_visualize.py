import matplotlib.pylab as plt
import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

df = pd.read_csv("../data/podatki/ratings.csv")
ax = df.score.plot.hist(bins=20)

# df = pd.read_csv("../data/MovieLens/ml-latest/ratings.csv")
# ax = df.rating.plot.hist(bins=10)

plt.show()
