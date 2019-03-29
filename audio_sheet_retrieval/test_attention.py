"""
Visualize Attention model on the test data.

Example Call:
    python test_attention.py --model models/mutopia_ccal_cont_rsz_dense_att.py \
     --data mutopia --train_split ../../msmd/msmd/splits/all_split.yaml \
     --config exp_configs/mutopia_full_aug_lc.yaml --n_test 1000
"""
from __future__ import print_function

import os

try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import yaml
import lasagne
import theano
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model, compile_tag
from audio_sheet_retrieval.utils.batch_iterators import batch_compute1
from audio_sheet_retrieval.utils.data_pools import AudioScoreRetrievalPool
from audio_sheet_retrieval.utils.mutopia_data import load_piece_list


def entropy(x):
    x += 1e-9
    log_prob = np.log(x)
    return -np.sum(log_prob * x)


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Evaluate cross-modality retrieval model.')
    parser.add_argument('--model', help='select model to evaluate.')
    parser.add_argument('--data', help='select evaluation data.', type=str)
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--n_test', help='number of test samples used.', type=int, default=None)
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    parser.add_argument('--seed', help='query direction.', type=int, default=23)
    args = parser.parse_args()

    # load config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)

    # load split
    with open(args.train_split, 'rb') as hdl:
        split = yaml.load(hdl)

    # select model
    model, _ = select_model(args.model)

    if not hasattr(model, 'prepare'):
        model.prepare = None

    # select data
    print('Loading data...')
    eval_set = 'test'
    te_images, te_specs, te_o2c_maps, te_audio_pathes = load_piece_list(split[eval_set], fps=config['FPS'])
    te_pool = AudioScoreRetrievalPool(te_images, te_specs, te_o2c_maps, te_audio_pathes,
                                      spec_context=config['SPEC_CONTEXT'], sheet_context=config['SHEET_CONTEXT'],
                                      staff_height=config['STAFF_HEIGHT'], shuffle=False,
                                      return_piece_names=True, return_n_onsets=True)

    print('Building network %s ...' % model.EXP_NAME)
    layers = model.build_model(input_shape_1=[1, te_pool.staff_height, te_pool.sheet_context],
                               input_shape_2=[1, te_pool.spec_bins, te_pool.spec_context],
                               show_model=False)

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print('Experimental Tag:', tag)

    # load model parameters
    if args.estimate_UV:
        model.EXP_NAME += '_est_UV'
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if tag is None else 'params_%s.pkl' % tag
    dump_file = os.path.join(out_path, dump_file)

    print('\n')
    print('Loading model parameters from:', dump_file)
    with open(dump_file, 'rb') as fp:
        params = pickle.load(fp)
    if isinstance(params[0], list):
        # old redundant dump
        for i_layer, layer in enumerate(layers):
            lasagne.layers.set_all_param_values(layer, params[i_layer])
    else:
        # non-redundant dump
        lasagne.layers.set_all_param_values(layers, params)

    print('\nCompiling prediction functions...')
    l_view1, l_view2, l_v1latent, l_v2latent = layers

    # audio attention layer
    attention_layer = l_v2latent
    while attention_layer.name != 'attention':
        if not hasattr(attention_layer, 'input_layer'):
            attention_layer = attention_layer.input_layers[1]
        else:
            attention_layer = attention_layer.input_layer

    compute_attention = theano.function(inputs=[l_view2.input_var],
                                        outputs=lasagne.layers.get_output(attention_layer, deterministic=True))

    # iterate test data
    print('Computing attention...')

    # compute output on test set
    n_test = args.n_test if args.n_test is not None else te_pool.shape[0]
    indices = np.linspace(0, te_pool.shape[0] - 1, n_test).astype(np.int)
    X1, X2, piece_names, n_onsets = te_pool[indices]

    # compute attention
    prepare = getattr(model, 'prepare', None)

    # compute attention masks
    att = batch_compute1(X2, compute_attention, batch_size=100, verbose=False, prepare=None)
    max_attention = np.max(att)

    # get entropy for each attention mask
    entropies = [entropy(a) for a in att]

    # pieces = []
    # for cur_piece_idx, cur_piece_name in enumerate(piece_names):
    #     cur_piece = dict()
    #     cur_piece['name'] = cur_piece_name
    #     cur_piece['entropy'] = entropies[cur_piece_idx]
    #     pieces.append(cur_piece)
    # import pandas as pd
    # pieces = pd.DataFrame(pieces)

    # sort by entropy
    sorted_idxs = np.argsort(entropies)
    entropies = np.array(entropies)[sorted_idxs]
    X1 = X1[sorted_idxs]
    X2 = X2[sorted_idxs]
    att = att[sorted_idxs]
    piece_names = [piece_names[cur_piece_idx] for cur_piece_idx in sorted_idxs.astype(int).tolist()]
    n_onsets = np.array(n_onsets)[sorted_idxs]

    # apply attention to spectrogram
    X2_att = X2 * att[:, np.newaxis, np.newaxis]

    plot_rows = 3
    plot_cols = 5

    plt.figure('AttentionSamples', figsize=(30, 5 * plot_rows))
    gs = gridspec.GridSpec(plot_rows, plot_cols, height_ratios=[3, 2, 2])
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=1.0, wspace=0.05, hspace=0.01)
    idxs = np.linspace(0, att.shape[0] - 1, plot_cols).astype(np.int)
    for i, idx in enumerate(idxs):
        row_idx = 2 * (i // plot_cols)
        col_idx = np.mod(i, plot_cols)
        print(i, piece_names[idx])
        ax0 = plt.subplot(gs[row_idx + 0, col_idx])
        ax0.imshow(X1[idx, 0], cmap='gray')
        ax0.set_xticks([], [])
        ax0.set_yticks([], [])
        ax0.set_xlabel('%d Pixel' % X1.shape[-1], fontsize=36)

        a = att[idx]
        ax1 = plt.subplot(gs[row_idx + 1, col_idx])
        ax1.plot(a, 'k-', linewidth=4, alpha=0.85)
        ax1.fill_between(range(att.shape[1]), 0, a, facecolors='gray', alpha=0.5)
        ax1.set_xlim([0, att.shape[1]])
        ax1.set_ylim([0, max_attention])
        ax1.axis('off')

        ax2 = plt.subplot(gs[row_idx + 2, col_idx])
        ax2.imshow(X2[idx, 0], cmap='binary', origin='lower', aspect='auto')
        ax2.set_xticks([], [])
        ax2.set_yticks([], [])
        ax2.set_xlabel('%d Frames' % att.shape[1], fontsize=36)

        # axes[row_idx+2, col_idx].imshow(X2_att[idx, 0], cmap='viridis', origin='lower', aspect='auto')
    plt.savefig('AttentionSamples.png')

    plt.figure('Attention Mask', figsize=(20, 10))
    plt.clf()
    plt.plot(att.T, 'k-', linewidth=1.0, alpha=0.2)
    mean_attention = att.mean(axis=0)
    plt.plot(mean_attention, 'w-', linewidth=5.0, alpha=1.0)
    plt.grid()
    plt.xlim([-1, att.shape[1]])
    plt.title('Mean Entropy: %.5f' % entropy(mean_attention))
    plt.tight_layout()
    plt.savefig('AttentionMasks.png')

    fig = plt.figure('Notes vs. Entropy', figsize=(12, 6))
    plot = fig.add_subplot(111)
    plot.tick_params(axis='both', which='major', labelsize=16)
    plt.clf()
    plt.scatter(n_onsets, entropies, alpha=0.2)
    plt.xlabel('#Onsets in Audio Frame', fontsize=18)
    plt.ylabel('Entropy', fontsize=18)
    plt.tight_layout()
    plt.savefig('AttentionNoteEntropy.png')

    print(np.around(mean_attention, 2))

    # # compute t-SNE embedding of attention maps
    # from sklearn.manifold import TSNE
    # e = TSNE(n_components=2).fit_transform(att)
    #
    # plt.figure('Attention Mask', figsize=(20, 20))
    # plt.clf()
    # plt.plot(e[:, 1], e[:, 0], 'o')
    # plt.title('Attention t-SNE')
    # plt.savefig('AttentionEmbedding.png')
    #
    # # prepare image
    # scale = 50
    # dr = X2.shape[2]
    # dc = X2.shape[3]
    # e += np.abs(e.min(axis=0, keepdims=True))
    # e /= e.max(axis=0, keepdims=True)
    # max_r, max_c = np.max(e, axis=0)
    # e *= (scale * dr)
    # e += dr
    #
    # dim_c = int(max_c * (scale * dc) + 2 * dc)
    # dim_r = int(max_r * (scale * dr) + 2 * dr)
    # I = np.zeros((dim_r, dim_c), dtype=np.float32)
    #
    # for i in xrange(e.shape[0]):
    #     rc, cc = e[i, :]
    #     r0 = int(rc - dr // 2)
    #     r1 = r0 + dr
    #     c0 = int(cc - dc // 2)
    #     c1 = c0 + dc
    #
    #     Sample = X2[i, 0]
    #     Sample[:, 0] = 1
    #     Sample[:, -1] = 1
    #     Sample[0, :] = 1
    #     Sample[-1, :] = 1
    #     I[r0:r1, c0:c1] = Sample
    #
    # plt.figure('Attention Miniatures', figsize=(40, 40))
    # plt.clf()
    # plt.imshow(I, cmap='viridis', origin='lower')
    # plt.savefig('AttentionMiniatures.png')
