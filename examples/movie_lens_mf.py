from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.parallel import MultiSimulator
from pyrec.sims.rand import RandomFromTopNSimulator
from pyrec.plots import multi_plot, plot_ratings_violin, multi_success_stops
from pyrec.recs.mf import MatrixFactorization
from pyrec.recs.post import UnbiasedMatrixFactorization, \
    UnbiasedUsersMatrixFactorization, UserWantMatrixFactorization


RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)
figure_file = "../figures/ml_inv_mf"

sims = []
inv = Inventory(uir_data)
mf = MatrixFactorization(max_iteration=200, batch_size=100)
mf.fit(uir_data)
sims.append(RandomFromTopNSimulator(f"mf", uir_data, mf, inv))

inv = Inventory(uir_data)
umf = UnbiasedMatrixFactorization(max_iteration=200, batch_size=100)
umf.fit(uir_data)
sims.append(RandomFromTopNSimulator(f"umf", uir_data, umf, inv))

inv = Inventory(uir_data)
uumf = UnbiasedUsersMatrixFactorization(max_iteration=200, batch_size=100)
uumf.fit(uir_data)
sims.append(RandomFromTopNSimulator(f"uumf", uir_data, uumf, inv))

inv = Inventory(uir_data)
uwmf = UserWantMatrixFactorization(max_iteration=200, batch_size=100)
uwmf.fit(uir_data)
sims.append(RandomFromTopNSimulator(f"uwmf", uir_data, uwmf, inv))


ms = MultiSimulator(20_000)
ms.set_sims(sims)
ms.run_parallel()


plot_ratings_violin(sims[0], save_file=figure_file + "_rviola_mf.png")
plot_ratings_violin(sims[0], save_file=figure_file + "_rviola_umf.png")
plot_ratings_violin(sims[0], save_file=figure_file + "_rviola_uumf.png")
plot_ratings_violin(sims[0], save_file=figure_file + "_rviola_uwmf.png")

multi_plot(sims, data="sold_i", save_file=figure_file + "_sold.png")
multi_success_stops(sims, list(range(1_000, 20_001, 1_000)),
                    save_file=figure_file + "_20k_success_step.png")
