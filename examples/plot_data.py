import numpy as np
import matplotlib.pyplot as plt

from pyrec.data import UIRData
from pyrec.inventory import Inventory


####################
# trgovski podatki #
####################
uir_data = UIRData.from_csv("../data/podatki/ratings.csv")

print("Trgovski podatki")
print(f"users: {uir_data.n}, items: {uir_data.m}, ratings: {uir_data.uir_n}")

plt.hist(uir_data.raw_data.ratings, bins=10)
plt.title('Porazdelitev ocen Trgovskih podatkov')
plt.xlabel("Ocene")
plt.ylabel("Frekvenca")
plt.show()

inv = Inventory.from_csv("../data/podatki/inv.csv")
print(f"Količina zaloge: {inv.start_size}")

counts = np.sort(inv.counts)[::-1]
print(counts[:10])

plt.plot(counts)
plt.title('Porazdelitev zaloge za Trgovske podatke')
plt.xlabel("Rank izdelka")
plt.ylabel("Število enot na zalogi")
plt.show()


#############
# MovieLens #
#############
uir_data = UIRData.from_csv("../data/MovieLens/ml-1m/ratings.csv")

print("MovieLens 1m")
print(f"users: {uir_data.n}, items: {uir_data.m}, ratings: {uir_data.uir_n}")

plt.hist(uir_data.raw_data.ratings, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.title('Porazdelitev ocen MovieLens 1m')
plt.xlabel("Ocene")
plt.ylabel("Frekvenca")
plt.show()

inv = Inventory(uir_data)
print(f"Količina zaloge: {inv.start_size}")

counts = np.sort(inv.counts)[::-1]
print(counts[:10])
plt.plot(counts)
plt.title('Porazdelitev zaloge za MovieLens 1m')
plt.xlabel("Rank izdelka")
plt.ylabel("Število enot na zalogi")
plt.show()