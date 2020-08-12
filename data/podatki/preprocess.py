import csv
import pandas as pd


df = pd.read_csv("nakupi.csv", parse_dates=[2])

item_user_count = df.groupby(['gn', 'artikel'], as_index=False).sum()[['gn', 'artikel', 'kolicina']]
item_min_count = item_user_count.groupby(['artikel']).min()
item_min_count[item_min_count.kolicina < 0] = 0
item_max_count = item_user_count.groupby(['artikel']).max()

item_user_time = df.groupby(['gn', 'artikel'], as_index=False).max()[['gn', 'artikel', 'datum_prometa']]
item_min_time = df.groupby(['artikel']).min()
item_max_time = df.groupby(['artikel']).max()


def time2rating(item, time_obj):
    min_time = item_min_time.loc[item, 'datum_prometa']
    max_time = item_max_time.loc[item, 'datum_prometa']
    if (max_time - min_time) == pd.Timedelta(0):
        return 1
    return (time_obj - min_time) / (max_time - min_time)


def count2rating(item, count):
    min_count = item_min_count.loc[item, 'kolicina']
    max_count = item_max_count.loc[item, 'kolicina']
    if (max_count - min_count) == 0:
        return 1
    return (count - min_count) / (max_count - min_count)


res = [['gn', 'artikel', 'rating']]
for c, t in zip(item_user_count.iterrows(), item_user_time.iterrows()):
    assert c[1]['gn'] == t[1]['gn'] and c[1]['artikel'] == t[1]['artikel']
    item = c[1]['artikel']
    t_rating = time2rating(item, t[1]['datum_prometa'])
    c_rating = count2rating(item, c[1]['kolicina'])
    res.append([int(c[1]['gn']), int(c[1]['artikel']), t_rating + c_rating])


with open("ratings.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(res)
