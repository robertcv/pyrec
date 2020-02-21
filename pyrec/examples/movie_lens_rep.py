from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.simulator import RandomFromTopNSimulator, RepeatedSimulation, \
    multi_success_err
from pyrec.recommender import MatrixFactorization, WeightedRecommender


RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)
inv = Inventory(uir_data)

rec_kwargs = {"inv": None, "verbose": False,
              "rec": MatrixFactorization, "rec_kwargs": {"max_iteration": 10,
                                                         "batch_size": 100}}

sims = []
alphas = [0, 0.7, 1]
for a in alphas:
    rec_kwargs["alpha"] = a
    rs = RepeatedSimulation(f"a={a}", uir_data, inv,
                            WeightedRecommender, rec_kwargs,
                            RandomFromTopNSimulator, {"verbose": False})
    rs.run(10_000, 3)
    sims.append(rs)

figure_file = "../../figures/ml_inv"
multi_success_err(sims)
