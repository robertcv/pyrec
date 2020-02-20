from copy import deepcopy

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender
from pyrec.simulator import BaseSimulator
from pyrec.parallel import MultiSimulator, MultiRecommender


class RepeatedSimulation:
    def __init__(self, name, data: UIRData, inv: Inventory,
                 rec, rec_kwargs: dict, sim, sim_kwargs: dict):
        """data and inv must be an instance, others are just class"""
        self.name = name
        self.data = data
        self.inv = inv
        self.rec = rec
        self.rec_kwargs = rec_kwargs
        self.sim = sim
        self.sim_kwargs = sim_kwargs

        self.sim_data = {}

    def run(self, n, rep):
        print(f"Start repeated Simulation {self.name}")

        data = [self.data.copy() for _ in range(rep)]
        inv = [self.inv.copy() for _ in range(rep)]

        recs = []
        for i in range(rep):
            if "inv" in self.rec_kwargs:
                self.rec_kwargs["inv"] = inv[i]
            rec = self.rec(**self.rec_kwargs)  # type: BaseRecommender
            recs.append(rec)

        mr = MultiRecommender("../../tmp")
        mr.set_recs(recs, data)
        mr.fit_parallel()

        print(f"finished fitting models")

        sims = []
        for i in range(rep):
            self.sim_kwargs["name"] = ""
            self.sim_kwargs["data"] = data[i]
            self.sim_kwargs["inv"] = inv[i]
            self.sim_kwargs["rec"] = recs[i]
            sim = self.sim(**self.sim_kwargs)  # type: BaseSimulator
            sims.append(sim)

        ms = MultiSimulator(n)
        ms.set_sims(sims)
        ms.run_parallel()

        self.sim_data = {
            "empty_items": [sim.sim_data["empty_items"][-1] for sim in sims],
            "sold_items": [sim.sim_data["sold_items"][-1] for sim in sims],
            "not_sold_items": [sim.sim_data["not_sold_items"][-1] for sim in sims],
            "rmse": [sim.sim_data["rmse"] for sim in sims],
        }


if __name__ == '__main__':
    from pyrec.recommender import MatrixFactorization, WeightedRecommender
    from pyrec.simulator import RandomFromTopNSimulator

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    rec_kwargs = {"alpha": 0.5, "inv": None, "verbose": False,
                  "rec": MatrixFactorization,
                  "rec_kwargs": {}}

    rs = RepeatedSimulation("test", uir_data, inv,
                            WeightedRecommender, rec_kwargs,
                            RandomFromTopNSimulator, {"verbose": False})
    rs.run(1_000, 5)
