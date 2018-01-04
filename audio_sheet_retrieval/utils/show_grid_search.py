# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import numpy as np

from plotting import BColors
from grid_search import GridSearch

col = BColors()


if __name__ == '__main__':
    """
    Present results of grid search
    """

    # add argument parser
    parser = argparse.ArgumentParser(description='Show evaluation plot.')
    parser.add_argument('--file', help='grid_search.pkl file.')
    parser.add_argument('--topk', help='show top k examples.', type=int, default=10)
    parser.add_argument('--show', help='show grid search plots.', action='store_true')
    args = parser.parse_args()
    
    # load grid search file
    print("Loading file", args.file)
    gs = GridSearch.load(args.file)
    
    # print statistics
    stats = gs.get_stats()
    print("")
    print("Stats:")
    print(" - mean: %.5f, med: %.5f, std: %.5f" % (stats['mean'], stats['median'], stats['std']))
    print(" - min:  %.5f, max: %.5f" % (stats['min'], stats['max']))
    print(" - eval: %d, total: %d" % (stats['n_eval'], stats['n_total']))
    
    # print best configuration
    print("")
    gs.print_best()
    
    # print best configuration
    print("")
    gs.print_worst()
    
    # print top 10 configuration
    top_k = gs.get_top_k(k=args.topk)
    prev_worst = None
    
    print("\nTop %d:" % args.topk)
    for i in xrange(args.topk):
        value, combination, values = top_k[i]
        print("\n%02d: %.5f | " % (i, value), end="")

        for v in values:
            txt = "%f" % v
            if v < prev_worst and prev_worst is not None:
                txt = col.print_colored(txt, col.WARNING)
            print(txt, end=", ")
        print("")

        for k, v in combination.iteritems():
            print(k, v, end=", ")
        print("")

        prev_worst = np.max(values)
    
    # visualize search grid
    if args.show:
        gs.visualize_grid()
