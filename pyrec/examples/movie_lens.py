from pyrec.data import UIRData
from pyrec.inventory import Inventory, UniformInventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender, \
    WeightedRecommender, MostInInvStaticRecommender
from pyrec.simulator import RandomFromTopNSimulator, \
    plot_ratings_violin, multi_plot
from pyrec.parallel import MultiSimulator


# read data from file
print("load data")
RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)
print(uir_data)

# create inventory for simulations
UNIFORM = False
if UNIFORM:
    inv = UniformInventory(uir_data)
else:
    inv = Inventory(uir_data)
print(inv)

# fit models and create sims
print("fit models")
_inv = inv.copy()
mf = MatrixFactorization.load_static("../../models/ml-small-mf")
mf.data = uir_data
sim1 = RandomFromTopNSimulator("mf", uir_data, mf, _inv)

_inv = inv.copy()
wr = WeightedRecommender(0.85, _inv, mf, {})
wr.fit(uir_data)
sim3 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
wr = WeightedRecommender(0.75, _inv, mf, {})
wr.fit(uir_data)
sim5 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
miir = MostInInvRecommender(_inv)
miir.fit(uir_data)
sim7 = RandomFromTopNSimulator("inv", uir_data, miir, _inv)

_inv = inv.copy()
miisr = MostInInvStaticRecommender(_inv)
miisr.fit(uir_data)
sim8 = RandomFromTopNSimulator("inv s", uir_data, miisr, _inv)

# run simulations
print("run simulations")
ms = MultiSimulator(50_000)
ms.set_sims([sim1, sim3, sim5, sim7, sim8])
ms.run_parallel()

# plot data
print("plot data")

figure_file = "../../figures/ml"
if UNIFORM:
    figure_file += "_uinv"
else:
    figure_file += "_inv"

plot_ratings_violin(sim1, save_file=figure_file + "_rviola_mf.png")
plot_ratings_violin(sim3, save_file=figure_file + "_rviola_alpha_85.png")
plot_ratings_violin(sim5, save_file=figure_file + "_rviola_alpha_75.png")
plot_ratings_violin(sim7, save_file=figure_file + "_rviola_mii.png")
plot_ratings_violin(sim8, save_file=figure_file + "_rviola_miis.png")

# multi_plot([sim1, sim3, sim5, sim7, sim8],
#            data="sold_i",
#            save_file=figure_file + "_sold.png")
# multi_plot([sim1, sim3, sim5, sim7, sim8],
#            data="empty_i",
#            save_file=figure_file + "_empty.png")
# multi_plot([sim1, sim3, sim5, sim7, sim8],
#            data="not_sold_i",
#            save_file=figure_file + "_not_sold.png")
