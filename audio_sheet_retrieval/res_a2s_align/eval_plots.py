
import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


if __name__ == '__main__':
    """ main """

    res_files = glob.glob("*.pkl")
    n_methods = len(res_files)

    per_method_results = dict()

    for res_file in res_files:
        print ""
        print res_file

        # load results
        with open(res_file, "rb") as fp:
            piece_pxl_errors = pickle.load(fp)

        # collect errors
        all_pxl_errors = []
        for piece, errs in piece_pxl_errors.iteritems():
            all_pxl_errors.extend(errs)

        all_pxl_errors = np.array(all_pxl_errors)
        abs_pxl_errors = np.abs(all_pxl_errors)
        abs_pxl_errors /= 835

        # report results for piece
        print "Mean Error:   %.3f" % np.mean(abs_pxl_errors)
        print "Median Error: %.3f" % np.median(abs_pxl_errors)
        print "Std Error:    %.3f" % np.std(abs_pxl_errors)

        # keep per piece errors
        method = res_file.rsplit('_')[-1].split('.pkl')[0]
        per_method_results[method] = abs_pxl_errors
        # per_method_results[method] = all_pxl_errors

    # initialize data frame
    methods = []
    abs_pxl_errors = []
    for k, v in per_method_results.iteritems():
        methods += [k] * len(v)
        abs_pxl_errors.append(v)

    abs_pxl_errors = np.asarray(abs_pxl_errors).T

    # fix labels
    methods = np.asarray(methods)
    methods[methods == 'baseline'] = 'Linear'
    methods[methods == 'pydtw'] = 'DTW'

    df = pd.DataFrame()
    df['Alignment Method'] = methods
    df['|Alignment Error|'] = np.concatenate([abs_pxl_errors[:, i] for i in xrange(n_methods)])
    # figures
    sns.set(style="ticks", palette="muted", color_codes=True)

    plt.figure(figsize=(7, 3))
    plt.clf()

    ax = sns.boxplot(x="|Alignment Error|", y="Alignment Method", data=df,
                     whis=np.inf, color="b")

    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        tick.label.set_rotation(45)

    plt.subplots_adjust(left=0.18, bottom=0.25)

    plt.show(block=True)
