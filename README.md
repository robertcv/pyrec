## PyRec

This module holds code for my master's thesis. The goal is to create
a recommender system that not only learns user item preferences but
also takes item inventory into account.

This is intended to be used as a sandbox for testing different
recommendation systems and evaluating their performance by simulating users
buying items from inventory.

The major classes are:
* [UIRData](pyrec/data.py) - used to store and preprocess data
* [Inventory](pyrec/inventory.py) - hold inventory data
* [BaseRecommender](pyrec/recs/base.py) - base class for recommenders
* [MatrixFactorization](pyrec/recs/mf.py) - implementation of matrix factorization recommender system
* [BaseSimulator](pyrec/sims/base.py) - base class for simulations

Other interesting functionalities are running repeated simulations on
different initializations, training recommender systems and simulating them in
parallel ([works only on Unix systems](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)),
diverse plotting functions, more recommender systems.

A simple use case would be:
```python
from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recs.mf import MatrixFactorization
from pyrec.sims.rand import BestSimulator

# load data from csv, preprocess, split into train, validation and test
RATINGS_FILE = "data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)

# initialize matrix factorization, train the model
mf = MatrixFactorization()
mf.fit(uir_data)

# create inventory from rating frequency if no explicit inventory data is given
inv = Inventory(uir_data)

# initialize and run the simulation
sim = BestSimulator("rand", uir_data, mf, inv)
results = sim.run()

print(results)
```
For more examples see the [examples](examples) folder.
