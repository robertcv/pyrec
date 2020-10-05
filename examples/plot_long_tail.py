import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


x = [0, 60, 150, 185, 233,
     297, 391, 513, 783, 1_040,
     1_552, 2_500, 3_000, 4_000, 5_000,
     7_500, 10_000, 15_000, 20_000, 25_000]

y = [200_000, 180_000, 160_000, 140_000, 120_000,
     100_000, 80_000, 60_000, 40_000, 30_000,
     20_000, 12_064, 10_788, 9212, 8820,
     8_036, 7_448, 6664, 5488, 4704]

f = interpolate.interp1d(x, y, kind='quadratic')
xnew = np.arange(0, 25_000, 10)
ynew = f(xnew)

plt.plot(xnew, ynew)
plt.xlabel("Rank pesmi")
plt.ylabel("Število prenosov")
plt.xticks(np.arange(0, 25_001, 5_000), [f"{int(i / 1000)}.000" if i else 0 for i in np.arange(0, 25_001, 5_000)])
plt.yticks(np.arange(0, 200_001, 20_000), [f"{int(i / 1000)}.000" if i else 0 for i in np.arange(0, 200_001, 20_000)])
plt.show()
