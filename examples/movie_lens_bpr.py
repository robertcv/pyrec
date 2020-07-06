from pyrec.data import UIRData
from pyrec.inventory import Inventory, RandomInventory, UniformInventory
from pyrec.sims.rand import BestSimulator
from pyrec.sims.repeated import RepeatedSimulation
from pyrec.parallel import MultiSimulator
from pyrec.plots import multi_success_stops, multi_success_err, plot_ratings_dist
from pyrec.recs.mf import MatrixFactorization, NNMatrixFactorization, \
    RandomMatrixFactorization
from pyrec.recs.bpr import BPR, UnbiasedBPR
from pyrec.recs.post import UnbiasedMatrixFactorization, \
    UserNotWantMatrixFactorization
from pyrec.recs.inv import MostInInvStaticRecommender, MostInInvRecommender
from pyrec.recs.weighted import WeightedRecommender


RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)
figure_file = "../figures/bpr"

k = 30
n = 20_000

sim_kwargs = {"verbose": True}

recs = [
    ("mf", MatrixFactorization),
    ("nnmf", NNMatrixFactorization),
    ("umf", UnbiasedMatrixFactorization),
    ("mii", MostInInvRecommender),
    ("wrs", WeightedRecommender),
]

sims = []

inv = Inventory(uir_data)
r = MatrixFactorization(max_iteration=200, batch_size=100)
r.fit(uir_data)
sims.append(BestSimulator("mf", uir_data, r, inv))

inv = Inventory(uir_data)
r = NNMatrixFactorization(max_iteration=200, batch_size=100)
r.fit(uir_data)
sims.append(BestSimulator("nnmf", uir_data, r, inv))

inv = Inventory(uir_data)
r = MostInInvRecommender(inv=inv)
r.fit(uir_data)
sims.append(BestSimulator("mii", uir_data, r, inv))

inv = Inventory(uir_data)
r = BPR()
r.fit(uir_data)
sims.append(BestSimulator("bpr", uir_data, r, inv))

inv = Inventory(uir_data)
r = UnbiasedBPR()
r.fit(uir_data)
sims.append(BestSimulator("ubpr", uir_data, r, inv))

# run simulations
print("run simulations")
ms = MultiSimulator(n)
ms.set_sims(sims)
ms.run_parallel()

multi_success_stops(sims, list(range(1_000, n + 1, 500)),
                    save_file=figure_file + "_best_20k_success_step.png")

# repeated

# rec_kwargs["rec2_kwargs"]["inv"] = inv
# rec_kwargs["inv"] = inv
# rec_kwargs["verbose"] = False
# sim_kwargs["verbose"] = False
#
# sims = []
# for name, rec in recs:
#     rs = RepeatedSimulation(name, uir_data, inv,
#                             rec, rec_kwargs,
#                             BestSimulator, sim_kwargs)
#     rs.run(n, k)
#     sims.append(rs)
#
# multi_success_err(sims, save_file=figure_file + "_best_20k_success_err.png")
