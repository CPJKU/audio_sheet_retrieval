"""
Script to compute the attention models and embedding spaces on test data for a given tempo, and save corresponding
variables to pickle file.
The variables are saved to utils/visualizations/variables_tag.pkl, where tag corresponds to the experiment label,
for example 'variables_mutopia_ccal_cont_rsz_dense_att_est_UV_all_split_full_aug_lc_1250'

Example Call:
    python get_att_emb.py --model models/mutopia_ccal_cont_rsz_dense_att.py \
     --data mutopia --train_split ../../msmd/msmd/splits/all_split.yaml \
     --config exp_configs/mutopia_full_aug_lc.yaml --estimate_UV --n_test 1000 --test_tempo 1.25
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
from audio_sheet_retrieval.utils.batch_iterators import batch_compute1, batch_compute2
from audio_sheet_retrieval.utils.data_pools import AudioScoreRetrievalPool, NO_AUGMENT
from audio_sheet_retrieval.utils.mutopia_data import load_piece_list
from audio_sheet_retrieval.utils.train_dcca_pool import eval_retrieval


def entropy(x):
    x += 1e-9
    log_prob = np.log(x)
    return -np.sum(log_prob * x, axis=x.ndim - 1)


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
    parser.add_argument('--dump_variables', help='dump variables for visualizations to utils/visualizations.',
                        action='store_true')
    parser.add_argument('--test_tempo', help='select different tempo ratio for testing (overwrites exp-config).',
                        type=float, default=1.0)
    args = parser.parse_args()

    # load config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)

    test_augment = NO_AUGMENT.copy()
    if args.test_tempo != 1.0:
        config['TEST_TEMPO'] = args.test_tempo
        test_augment = NO_AUGMENT.copy()
        test_augment['synths'] = [config["TEST_SYNTH"]]
        test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

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
    te_images, te_specs, te_o2c_maps, te_audio_pathes = load_piece_list(split[eval_set], fps=config['FPS'],
                                                                        aug_config=test_augment)
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

    # embedding spaces layer
    input_1 = input_2 = [l_view1.input_var, l_view2.input_var]
    compute_v1_latent = theano.function(inputs=input_1,
                                        outputs=lasagne.layers.get_output(l_v1latent, deterministic=True))
    compute_v2_latent = theano.function(inputs=input_2,
                                        outputs=lasagne.layers.get_output(l_v2latent, deterministic=True))

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
    entropies = entropy(att)

    print("Computing embedding space...")
    lv1 = batch_compute2(X1, X2, compute_v1_latent, np.min([100, n_test]), prepare1=model.prepare)
    lv2 = batch_compute2(X1, X2, compute_v2_latent, np.min([100, n_test]), prepare1=model.prepare)
    lv1_cca = lv1
    lv2_cca = lv2

    # reset n_test
    n_test = lv1_cca.shape[0]

    print("Computing performance measures...")
    mean_rank_te, med_rank_te, dist_te, hit_rates, map, ranks = eval_retrieval(lv1_cca, lv2_cca, return_ranks=True)
    ranks = np.array(ranks)

    # sort by entropy (NO NEED TO SORT VARIABLES HERE)
    sorted_idxs = np.argsort(entropies)
    # entropies = entropies[sorted_idxs]
    # X1 = X1[sorted_idxs]
    # X2 = X2[sorted_idxs]
    # att = att[sorted_idxs]
    # piece_names = [piece_names[cur_piece_idx] for cur_piece_idx in sorted_idxs.astype(int).tolist()]
    # n_onsets = np.array(n_onsets)[sorted_idxs]
    # ranks = np.array(ranks)[sorted_idxs]
    # lv1_cca = lv1_cca[sorted_idxs]
    # lv2_cca = lv2_cca[sorted_idxs]

    # Dumping the variables for visualizations:
    if args.dump_variables:
        ranks = np.array(ranks)
        n_onsets = np.array(n_onsets)
        tag = compile_tag(args.train_split, args.config).replace('mutopia_', '')
        expname = 'variables_' + model.EXP_NAME + '_'
        # res_file = dump_file.replace("params_", "variables_").replace(".pkl", "")
        res_file = 'utils/visualizations/' + expname + tag + '_{}.pkl'.format(int(args.test_tempo * 1000))
        with open(res_file, 'wb') as fh:  # Python 3: open(..., 'wb')
            pickle.dump([entropies, X1, X2, att, piece_names, n_onsets, sorted_idxs, ranks, lv1_cca, lv2_cca], fh)

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
