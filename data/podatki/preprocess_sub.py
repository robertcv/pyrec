import pandas as pd


df = pd.read_csv("nakupi.csv")
df = df[:10000]
df.to_csv("nakupi_sub.csv", index=False)
