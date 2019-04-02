"""
Load variables from experiment and display an interactive data browser consisting of a scatter plot of
attention entropy vs number of note onsets in corresponding audio frame. By clicking on a point in the scatter plot,
the respective attention vector, spectrogram, and sheet music snippet will be displayed.

The variables are computed by running the script audio_sheet_retrieval/get_att_emb.py

Example call:
    python vis_browser.py --data variables_mutopia_ccal_cont_rsz_dense_att_est_UV_all_split_full_aug_lc_1250.pkl
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import cPickle as pickle
except ImportError:
    import pickle


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self):
        self.lastind = 0

        self.text = ax.text(0.05, 0.15, 'selected: none',
                            transform=ax.transAxes, va='top')
        self.ranktext = ax.text(0.05, 0.12, 'selected: none',
                                transform=ax.transAxes, va='top')
        self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=10, alpha=0.3,
                                 color='green', visible=False)

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
        self.update()

    def onpick(self, event):

        if event.artist != line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        ax1.cla()
        a = att[dataind]
        ax1.plot(a, 'k-', linewidth=4, alpha=0.85)
        ax1.fill_between(range(att.shape[1]), 0, a, facecolors='gray', alpha=0.5)
        ax1.set_xlim([0, att.shape[1]])
        ax1.set_ylim([0, np.max(att)])
        ax1.set_xticks([], [])
        # ax1.set_yticks([], [])
        ax1.set_ylabel('Attention')
        ax1.set_xlabel('%d frames' % att.shape[-1])  # , fontsize=18)
        # ax1.axis('off')

        ax2.cla()
        ax2.imshow(X2[dataind, 0], cmap='viridis', origin='lower', aspect='auto')
        ax2.set_xticks([], [])
        ax2.set_yticks([], [])
        ax2.set_xlabel('%d frames' % X2.shape[-1])  # , fontsize=18)

        ax3.cla()
        ax3.imshow(X1[dataind, 0], cmap='gray')
        ax3.set_xticks([], [])
        ax3.set_yticks([], [])
        ax3.set_xlabel('%d pixels' % X1.shape[-1])  # , fontsize=18)

        # ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
        #          transform=ax2.transAxes, va='top')
        # ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])

        self.text.set_text('selected: %d' % dataind)
        self.ranktext.set_text('corresponding rank: %d' % ranks[dataind])
        fig.canvas.draw()


if __name__ == '__main__':

    # add argument parser
    parser = argparse.ArgumentParser(description='Snippet browser visualization after evaluation.')
    parser.add_argument('--data', help='select visualization data.', type=str)
    arg = parser.parse_args()

    with open(arg.data, 'rb') as fh:
        entropies, X1, X2, att, piece_names, n_onsets, sorted_idxs, ranks, lv1_cca, lv2_cca = pickle.load(fh)

    # Plotting
    xs = n_onsets
    ys = entropies
    G = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
    fig = plt.figure(figsize=(15, 15))

    ax = plt.subplot(G[0, :])
    ax1 = plt.subplot(G[1, 0])
    ax2 = plt.subplot(G[2, 0])
    ax3 = plt.subplot(G[1:, 1])

    ax.set_title('click on point to plot corresponding spectrogram and sheet music snippet')
    ax.set_xlabel('#Onsets in audio frame')  # , fontsize=18)
    ax.set_ylabel('Entropy')  # , fontsize=18)

    # Colormap for the scatter plots accordings to the corresponding ranks
    # ranks = np.array(ranks)
    # sorted_ranks = ranks[sorted_idxs]
    ranks_inv = np.max(ranks) - ranks
    # sorted_ranks_inv = np.max(sorted_ranks) - sorted_ranks

    line = ax.scatter(xs, ys, cmap='magma', c=ranks_inv - 1, picker=5, alpha=0.4, s=50,
                      edgecolors='gray')  # 5 points tolerance

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    browser = PointBrowser()

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()
