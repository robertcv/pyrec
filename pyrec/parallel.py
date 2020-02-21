import os
from typing import List
from multiprocessing import Process, Manager, Queue, cpu_count

from pyrec.recommender import BaseRecommender
from pyrec.simulator import BaseSimulator
from pyrec.data import UIRData


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def run_sim(i: int, sim: BaseSimulator, n: int, q: Queue):
    sim_data = sim.run(n)
    q.put((i, sim_data))


class MultiSimulator:
    def __init__(self, n=1000):
        self.sims = []
        self.n = n

    def set_sims(self, sims: List['BaseSimulator']):
        self.sims = sims

    def run_parallel(self):
        max_at_once = cpu_count() // 3  # one sim normally uses 3 cors
        chunks = divide_chunks(list(range(len(self.sims))), max_at_once)
        for chunk in chunks:
            jobs = []
            q = Manager().Queue()
            for i in chunk:
                p = Process(target=run_sim,
                            args=(i, self.sims[i], self.n, q))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            while not q.empty():
                i, sim_data = q.get()
                self.sims[i].sim_data = sim_data


def fit_rec(rec: BaseRecommender, data: UIRData, dump_file: str):
    rec.fit(data)
    rec.save(dump_file)


class MultiRecommender:
    def __init__(self, dump_dir):
        self.dump_dir = dump_dir
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        self.recs = []
        self.data = []

    def set_recs(self, recs: List['BaseRecommender'], data: List['UIRData']):
        self.recs = recs
        self.data = data

    def fit_parallel(self):
        max_at_once = cpu_count() // 4  # one rec normally uses 4 cors
        chunks = divide_chunks(list(range(len(self.recs))), max_at_once)
        for chunk in chunks:
            jobs = []
            dumps = []
            for i in chunk:
                dump_file = os.path.join(self.dump_dir, f"dump_{i}.npz")
                p = Process(target=fit_rec,
                            args=(self.recs[i], self.data[i], dump_file))
                jobs.append(p)
                dumps.append((i, dump_file))
                p.start()

            for proc in jobs:
                proc.join()

            for i, df in dumps:
                self.recs[i].load(df)
                self.recs[i].data = self.data[i]


if __name__ == '__main__':
    from pyrec.recommender import MatrixFactorization

    RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    data = [uir_data.copy() for _ in range(1)]
    mfs = [MatrixFactorization(max_iteration=20, batch_size=100)
           for _ in range(1)]

    mr = MultiRecommender("../tmp")
    mr.set_recs(mfs, data)
    mr.fit_parallel()
    print(mr.recs[0]._predict(0, 0))
