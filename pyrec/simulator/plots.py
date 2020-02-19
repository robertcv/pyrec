from typing import List

import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cbook import violin_stats
import matplotlib.mlab as mlab

from pyrec.simulator import BaseSimulator, RepeatedSimulation


def plot_items(sim: BaseSimulator, save_file=None):
    empty_items = np.array(sim.sim_data["empty_items"])
    sold_items = np.array(sim.sim_data["sold_items"])

    fig, ax = plt.subplots()
    ax.plot(empty_items, label="empty items")
    ax.plot(sold_items, label="items sold")
    ax.legend()
    ax.set_xlabel('iteration')
    ax.set_ylabel('percent')
    ax.set_title(f'Inventory change throughout {sim.name} simulation')

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


def plot_ratings(sim: BaseSimulator, save_file=None, mean_size=50):
    test_ratings = np.array(sim.sim_data["test_ratings"])
    predicted_ratings = np.array(sim.sim_data["predicted_ratings"])

    if mean_size > 0:
        b = np.ones(mean_size) / mean_size
        b2 = np.ones(mean_size * 10) / (mean_size * 10)
        test_ratings = convolve(test_ratings, b, mode="same")
        predicted_ratings = convolve(predicted_ratings, b2, mode="same")

    fig, ax = plt.subplots()
    ax.plot(test_ratings, label="test rating")
    ax.plot(predicted_ratings, label="predicted rating")
    ax.legend()
    ax.set_xlabel('iteration')
    ax.set_ylabel('rating')
    ax.set_title(f'Change of rating for item in {sim.name} simulation')

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


def _violin_stats(data, n):
    def _kde_method(X, coords):
        # fallback gracefully if the vector contains only one value
        if np.all(X[0] == X):
            return (X[0] == coords).astype(float)
        kde = mlab.GaussianKDE(X)
        return kde.evaluate(coords)

    stats = []
    for i, d in enumerate(np.array_split(data, n)):
        _d = d[~np.isnan(d)]
        if len(_d) > 0:
            stats.extend(violin_stats(_d, _kde_method))
        else:
            stats.append(None)
    return stats


def _violin_plot(ax, data1, label1, data2, label2, n):
    stats1 = _violin_stats(data1, n)
    stats2 = _violin_stats(data2, n)

    width = 0.5
    m_width = 0.25
    alpha = 0.5
    fillcolor1 = ax._get_lines.get_next_color()
    fillcolor2 = ax._get_lines.get_next_color()

    # Render violins
    for pos, s1, s2 in zip(range(1, n + 1), stats1, stats2):
        if s1 is not None:
            coords = s1["coords"]
            val = np.array(s1['vals'])
            val = 0.5 * width * val / val.max()
            ax.fill_betweenx(coords, -val + pos, pos,
                             facecolor=fillcolor1, alpha=alpha)
            ax.hlines(s1['mean'], -m_width + pos, pos,
                      colors=fillcolor1)
        if s2 is not None:
            coords = s2["coords"]
            val = np.array(s2['vals'])
            val = 0.5 * width * val / val.max()
            ax.fill_betweenx(coords, pos, pos + val,
                             facecolor=fillcolor2, alpha=alpha)
            ax.hlines(s2['mean'], pos, pos + m_width,
                      colors=fillcolor2)

    ax.legend(handles=[
        mpatches.Patch(color=fillcolor1, alpha=alpha, label=label1),
        mpatches.Patch(color=fillcolor2, alpha=alpha, label=label2)
    ])


def plot_ratings_violin(sim: BaseSimulator, save_file=None, n=10):
    test_ratings = np.array(sim.sim_data["test_ratings"])
    predicted_ratings = np.array(sim.sim_data["predicted_ratings"])

    fig, ax = plt.subplots()
    _violin_plot(ax, test_ratings, "test ratings",
                 predicted_ratings, "predict ratings", n)

    ax.set_ylabel('rating')
    ax.set_xlabel('iteration')
    n_sub = int(len(test_ratings) / n)
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels(map(str, np.arange(1, n + 1) * n_sub))
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.set_title(f'Change of ratings in {sim.name} simulation')

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


def multi_plot(simulations: List['BaseSimulator'], data="empty_items",
               save_file=None):
    fig, ax = plt.subplots()

    for sim in simulations:
        ax.plot(sim.sim_data[data], label=sim.name)

    ax.legend()
    ax.set_xlabel('iteration')
    ax.set_title(f'Change of {data} throughout the simulations')

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


def multi_success(simulations: List['BaseSimulator'], save_file=None):
    rmse = [sim.sim_data["rmse"] for sim in simulations]
    sold = [sim.sim_data["sold_items"][-1] for sim in simulations]
    label = [sim.name[:6] for sim in simulations]

    fig, ax = plt.subplots()
    ax.scatter(rmse, sold)

    rmse_padding = min(rmse) * 0.01
    sold_padding = min(sold) * 0.01
    for i, txt in enumerate(label):
        ax.annotate(txt, (rmse[i] + rmse_padding, sold[i] + sold_padding))

    ax.set_xlabel('RMSE')
    ax.set_ylabel('Items sold [%]')
    ax.set_title(f'Success of RSs')

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


def multi_success_err(simulations: List['RepeatedSimulation'], save_file=None):
    rmse, sold = [], []
    rmse_e, sold_e = [], []
    label = []

    for s in simulations:
        rmse.append(np.mean(s.sim_data["rmse"]))
        rmse_e.append(np.std(s.sim_data["rmse"]))
        sold.append(np.mean(s.sim_data["sold_items"]))
        sold_e.append(np.std(s.sim_data["sold_items"]))
        label.append(s.name[:6])

    print(rmse)
    print(rmse_e)
    print(sold)
    print(sold_e)

    fig, ax = plt.subplots()
    ax.scatter(rmse, sold)
    ax.errorbar(rmse, sold, xerr=rmse_e, yerr=sold_e, fmt='none')

    rmse_padding = min(rmse) * 0.01
    sold_padding = min(sold) * 0.01
    for i, txt in enumerate(label):
        ax.annotate(txt, (rmse[i] + rmse_padding, sold[i] + sold_padding))

    ax.set_xlabel('RMSE')
    ax.set_ylabel('Items sold [%]')
    ax.set_title(f'Success of RSs')

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


if __name__ == '__main__':
    from pyrec.data import UIRData
    from pyrec.inventory import Inventory
    from pyrec.recommender import MatrixFactorization
    from pyrec.simulator import TestSimulator

    np.random.seed(0)

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    mf = MatrixFactorization.load_static("../../models/ml-small-mf")
    mf.data = uir_data
    sim = TestSimulator("best score", uir_data, mf, inv)
    sim.run(10_000)

    plot_ratings_violin(sim)
