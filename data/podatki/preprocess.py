import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

df = pd.read_csv("nakupi.csv", parse_dates=[2])
print('opened file')

# remove users with less then n purchases
user_doc_count = df.groupby(['gn']).agg({"dokument": "nunique"})
df = df.set_index('gn')
df = df.drop(index=user_doc_count[user_doc_count.dokument < 10].index)
df = df.reset_index()

# for each user, item, document sum up all item counts
user_item_doc_sum = df.groupby(['gn', 'artikel', 'dokument']).sum()[['kolicina']]
user_item_doc_sum.kolicina[user_item_doc_sum.kolicina < 0] = 0
# get average item count over all documents for each user
user_item = user_item_doc_sum.groupby(['gn', 'artikel']).mean()[['kolicina']]

# get min and max item count over all documents
item_doc_sum = df.groupby(['artikel', 'dokument'], as_index=False).sum()[['artikel', 'dokument', 'kolicina']]
item_doc_sum.kolicina[item_doc_sum.kolicina < 0] = 0
item_doc_min_count = item_doc_sum.groupby(['artikel']).min()[['kolicina']]
item_doc_max_count = item_doc_sum.groupby(['artikel']).max()[['kolicina']]

# calculate a score of how much of an item users bay compered to other users
user_item = user_item.join(item_doc_min_count, lsuffix='', rsuffix='_all_min')
user_item = user_item.join(item_doc_max_count, lsuffix='', rsuffix='_all_max')
user_item['item_count_score'] = (user_item['kolicina'] - user_item['kolicina_all_min']) / \
                                (user_item['kolicina_all_max'] - user_item['kolicina_all_min'])
user_item['item_count_score'] = user_item['item_count_score'].fillna(1)
print('score 1')

# get number of documents different documents for each user
user_doc_count = df.groupby(['gn']).agg({"dokument": "nunique"})
# get in how many different documents an item is bought
user_item_doc_count = df.groupby(['gn', 'artikel']).agg({"dokument": "nunique"})

# calculate a score on in how many different documents are items contained
user_item = user_item.join(user_doc_count)
user_item = user_item.join(user_item_doc_count, lsuffix='', rsuffix='_item')
user_item['item_doc_score'] = user_item['dokument_item'] / user_item['dokument']
print('score 2')

# sum up scores
user_item['score'] = user_item['item_count_score'] + user_item['item_doc_score']
index = user_item.index.copy()
user_item = user_item.reset_index()[['gn', 'artikel', 'score']]
# user_item = user_item[user_item['score'] != 0]


# normalize by user
def rank_normalization(x):
    # x_norm = rankdata(x) / len(x)
    x_norm = QuantileTransformer(n_quantiles=len(x)).fit_transform(
        x.values.reshape(-1, 1)).flatten()
    if isinstance(x, pd.Series):
        return pd.Series(x_norm, index=x.index)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    else:
        print(type(x))


user_item = user_item[['gn', 'score']].groupby(['gn']).transform(rank_normalization)
user_item.index = index
user_item = user_item[user_item['score'] != 0]
user_item = user_item.reset_index()
user_item.to_csv('ratings_norm.csv', index=False)
