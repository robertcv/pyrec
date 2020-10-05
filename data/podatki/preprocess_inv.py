import pandas as pd


df = pd.read_csv("ratings.csv")
df_inv = pd.read_csv("zaloge.csv")

df = df.set_index('artikel')
df_inv = df_inv[['artikel', 'kolicina']].groupby(['artikel']).mean()

item_subset = df_inv.index.unique().intersection(df.index.unique())

df = df.loc[item_subset]
df = df.reset_index()[['gn', 'artikel', 'score']]
df.to_csv('ratings.csv', index=False)

df_inv = df_inv.loc[item_subset]
df_inv = df_inv.reset_index()
df_inv = df_inv.round()
df_inv.to_csv('inv.csv', index=False)
