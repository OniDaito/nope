"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

stats.py - Perform stats on the C.elegans generated pickle file.

"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from elegans.result import Result


def correlate_results(results, thresh):
    ''' Correlate scores as best as we can, deciding which group is which.'''
    res = None
    
    for level in results:
        if level[0] == thresh:
            res = level[1]
            break

    og_asi = []
    og_asj = []
    g0 = []
    g1 = []
    g2 = []
    g3 = []

    for datum in res:
        if isinstance(datum, Result):
            og_asi.append(datum.og_asi_score)
            og_asj.append(datum.og_asj_score)
            g0.append(datum.new_scores[0])
            g1.append(datum.new_scores[1])
            g2.append(datum.new_scores[2])
            g3.append(datum.new_scores[3])
   
    og_asi = np.array(og_asi)
    og_asj = np.array(og_asj)
    g0 = np.array(g0)
    g1 = np.array(g1)
    g2 = np.array(g2)
    g3 = np.array(g3)

    print("Group Mean Scores (0-3)", np.mean(g0), np.mean(g1), np.mean(g2), np.mean(g3))

    print("First Pairing")
    gg = np.add(g0, g1)
    asi_combo_spear = spearmanr(og_asi, gg)
    asi_combo_pear = pearsonr(og_asi, gg)
    asj_combo_spear = spearmanr(og_asj, gg)
    asj_combo_pear = pearsonr(og_asj, gg)
    print("Group g0-g1 asi/asj", asi_combo_spear, asi_combo_pear, asj_combo_spear, asj_combo_pear)

    gg = np.add(g2, g3)
    asi_combo_spear = spearmanr(og_asi, gg)
    asi_combo_pear = pearsonr(og_asi, gg)
    asj_combo_spear = spearmanr(og_asj, gg)
    asj_combo_pear = pearsonr(og_asj, gg)
    print("Group g2-g3 asi/asj", asi_combo_spear, asi_combo_pear, asj_combo_spear, asj_combo_pear)

    print("Second Pairing")
    gg = np.add(g0, g2)
    asi_combo_spear = spearmanr(og_asi, gg)
    asi_combo_pear = pearsonr(og_asi, gg)
    asj_combo_spear = spearmanr(og_asj, gg)
    asj_combo_pear = pearsonr(og_asj, gg)
    print("Group g0-g2 asi/asj", asi_combo_spear, asi_combo_pear, asj_combo_spear, asj_combo_pear)

    gg = np.add(g1, g3)
    asi_combo_spear = spearmanr(og_asi, gg)
    asi_combo_pear = pearsonr(og_asi, gg)
    asj_combo_spear = spearmanr(og_asj, gg)
    asj_combo_pear = pearsonr(og_asj, gg)
    print("Group g1-g3 asi/asj", asi_combo_spear, asi_combo_pear, asj_combo_spear, asj_combo_pear)

    print("Third Pairing")
    gg = np.add(g0, g3)
    asi_combo_spear = spearmanr(og_asi, gg)
    asi_combo_pear = pearsonr(og_asi, gg)
    asj_combo_spear = spearmanr(og_asj, gg)
    asj_combo_pear = pearsonr(og_asj, gg)
    print("Group g0-g3 asi/asj", asi_combo_spear, asi_combo_pear, asj_combo_spear, asj_combo_pear)

    gg = np.add(g1, g2)
    asi_combo_spear = spearmanr(og_asi, gg)
    asi_combo_pear = pearsonr(og_asi, gg)
    asj_combo_spear = spearmanr(og_asj, gg)
    asj_combo_pear = pearsonr(og_asj, gg)
    print("Group g1-g2 asi/asj", asi_combo_spear, asi_combo_pear, asj_combo_spear, asj_combo_pear)


def graph_jaccard(results, threed=False):
    """ 
    Parse the results, looking for the best threshold,
    by graphing the jaccard.

    [threshold, [results]]
    """
    means = []
    stds = []
    threshes = []
    best = 0
    best_score = 0

    print("Num Samples:", len(results))

    for level in results:
        tres = level[1]
        jaccs = []
        threshes.append(level[0])

        for res in tres:
            if isinstance(res, Result):
                if threed:
                    jaccs.append(res.jacc3d)
                else:
                    jaccs.append(res.jacc2d)
            
        jaccs = np.array(jaccs)
        jaccs_mean = np.mean(jaccs)        

        if jaccs_mean > best_score:
            best_score = jaccs_mean
            best = level[0]

        means.append(jaccs_mean)
        stds.append(np.std(jaccs))

    # Plot the 2D jaccard scores
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x = threshes
    y = means
    err = stds
    yl = [y[i] - err[i] for i in range(len(y))]
    yu = [y[i] + err[i] for i in range(len(y))]
    ax.fill_between(x, yl, yu, step='pre', color='k', alpha=0.15)
    plt.errorbar(x, y, label='Jaccard Score (with Std.dev)')
    plt.legend(loc='lower right')    
    plt.xlabel('Threshold')
    plt.ylabel('Mean Jaccard Score')
    plt.show()

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOLLy C. Elegans prog.")
    parser.add_argument(
        "--pickle", default="", help="Process existing results from a pickle."
    )
    parser.add_argument(
        "--three", action="store_true", default=False, help="Do 3D masks and generate counts"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.0, help="Override best threshold",
    )

    args = parser.parse_args()

    if args.pickle != "" and os.path.exists(args.pickle):
        with open(args.pickle, 'rb') as f:
            results = pickle.load(f)
            best = graph_jaccard(results, args.three)
            threshes = [r[0] for r in results]
            print("Threshes", threshes)

            if args.thresh != 0 and args.thresh in threshes:
                best = args.thresh

            print("Best threshold", best)
            correlate_results(results, best)