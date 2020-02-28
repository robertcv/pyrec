from typing import Optional

import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender
from pyrec.simulator import BaseSimulator, sim_data
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

        self.sim_data = None  # type: Optional[sim_data]

    @staticmethod
    def _add_inv(inv, rec_kwargs: dict):
        if "inv" in rec_kwargs:
            rec_kwargs["inv"] = inv
        if "rec1_kwargs" in rec_kwargs and "inv" in rec_kwargs["rec1_kwargs"]:
            rec_kwargs["rec1_kwargs"]["inv"] = inv
        if "rec2_kwargs" in rec_kwargs and "inv" in rec_kwargs["rec2_kwargs"]:
            rec_kwargs["rec2_kwargs"]["inv"] = inv

    def run(self, n, rep):
        print(f"Start repeated Simulation {self.name}")

        data = [self.data.copy() for _ in range(rep)]
        inv = [self.inv.copy() for _ in range(rep)]

        recs = []
        for i in range(rep):
            self._add_inv(inv[i], self.rec_kwargs)
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

        self.sim_data = sim_data(
            empty_i=np.vstack([sim.sim_data.empty_i for sim in sims]),
            sold_i=np.vstack([sim.sim_data.sold_i for sim in sims]),
            not_sold_i=np.vstack([sim.sim_data.not_sold_i for sim in sims]),
            true_r=np.vstack([sim.sim_data.true_r for sim in sims]),
            pred_r=np.vstack([sim.sim_data.pred_r for sim in sims])
        )


if __name__ == '__main__':
    from pyrec.recommender import MatrixFactorization, WeightedRecommender, \
        MostInInvRecommender
    from pyrec.simulator import RandomFromTopNSimulator

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    rec_kwargs = {"alpha": 0.5, "verbose": False,
                  "rec1": MatrixFactorization,
                  "rec1_kwargs": {"max_iteration": 10, "batch_size": 100},
                  "rec2": MostInInvRecommender,
                  "rec2_kwargs": {"inv": None}}

    rs = RepeatedSimulation("test", uir_data, inv,
                            WeightedRecommender, rec_kwargs,
                            RandomFromTopNSimulator, {"verbose": False})
    rs.run(1_000, 5)
    print(rs.sim_data)
