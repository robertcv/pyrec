import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization
from implicit.als import AlternatingLeastSquares


RATINGS_FILE = "/home/robertcv/mag/data/MovieLens/ml-latest/ratings.csv"
# userId, movieId, rating, timestamp
data = pd.read_csv(RATINGS_FILE).values[:, :-1].astype(np.int32)

# remove users, items with less then 50 ratings
u_users, c_users = np.unique(data[:, 0], return_counts=True)
u_items, c_items = np.unique(data[:, 1], return_counts=True)

users = np.isin(data[:, 0], u_users[c_users > 50])
items = np.isin(data[:, 1], u_items[c_items > 50])

# it is not exact because of overlap but it it's a good approximation
useful = users & items
data = data[useful]

np.random.shuffle(data)
data[:, 2] = (data[:, 2] > 3).astype(int)

rows, row_pos = np.unique(data[:, 0], return_inverse=True)
cols, col_pos = np.unique(data[:, 1], return_inverse=True)
pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)

train_size = int(len(data) * 0.7)
train_row_pos, train_col_pos = row_pos[:train_size], col_pos[:train_size]
train_data = data[:train_size, 2]
test_row_pos, test_col_pos = row_pos[train_size:], col_pos[train_size:]
test_data = data[train_size:, 2]

pivot_table[train_row_pos, train_col_pos] = train_data


# W, H, n_iter = non_negative_factorization(pivot_table,
#                                           n_components=10,
#                                           alpha=0.001,
#                                           regularization="both",
#                                           verbose=1)


from scipy.sparse import csr_matrix

model = AlternatingLeastSquares(factors=10, iterations=200)
model.fit(csr_matrix(pivot_table))

pred_table = model.item_factors.dot(model.user_factors.T)
pred_test_data = pred_table[test_row_pos, test_col_pos]

rmse = np.sqrt(np.mean((pred_test_data - test_data)**2))
print(rmse)
