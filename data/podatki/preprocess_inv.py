import pandas as pd


df = pd.read_csv("ratings_norm.csv")
df_inv = pd.read_csv("zaloge.csv")
df_art = pd.read_csv("artikli.csv")
df_art = df_art[df_art["neto_kolicina"].astype('float') > 0.01]
df_art['artikel'] = df_art['artikel'].astype(int)

df = df.set_index('artikel')
df_art = df_art.set_index('artikel')
df_inv = df_inv[['artikel', 'kolicina']].groupby(['artikel']).mean()

item_subset = df_art.index.unique().intersection(df_inv.index.unique())
df_inv['kolicina'].loc[item_subset] = df_inv['kolicina'].loc[item_subset] / df_art['neto_kolicina'].loc[item_subset]

item_subset = df_inv.index.unique().intersection(df.index.unique())

df = df.loc[item_subset]
df = df.reset_index()[['gn', 'artikel', 'score']]
df.to_csv('ratings_inv_norm.csv', index=False)

df_inv = df_inv.loc[item_subset]
df_inv = df_inv.reset_index()
df_inv = df_inv.round()
df_inv.to_csv('inv_norm.csv', index=False)
