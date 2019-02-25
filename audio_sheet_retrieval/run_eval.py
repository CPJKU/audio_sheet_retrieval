#!/usr/bin/env python
# author: matthias dorfer

from __future__ import print_function

import os
import yaml
import six
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import lasagne
import theano
import numpy as np
import matplotlib.pyplot as plt

from config.settings import EXP_ROOT
from run_train import select_model, select_data, compile_tag

from utils.batch_iterators import batch_compute2

from utils.train_dcca_pool import eval_retrieval


def flip_variables(v1, v2):
    """ flip variables """
    tmp = v1.copy()
    v1 = v2
    v2 = tmp
    return v1, v2


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Evaluate cross-modality retrieval model.')
    parser.add_argument('--model', help='select model to evaluate.')
    parser.add_argument('--data', help='select evaluation data.', type=str)
    parser.add_argument('--show', help='show evaluation plots.', action='store_true')
    parser.add_argument('--n_test', help='number of test samples used.', type=int, default=None)
    parser.add_argument('--V2_to_V1', help='query direction.', action='store_true')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--max_dim', help='maximum dimension of retrieval space.', type=int, default=None)
    parser.add_argument('--seed', help='query direction.', type=int, default=23)
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    parser.add_argument('--dump_results', help='dump results of current run to file.', action='store_true')
    args = parser.parse_args()

    # select model
    model, _ = select_model(args.model)

    if not hasattr(model, 'prepare'):
        model.prepare = None

    print("Building network %s ..." % model.EXP_NAME)
    layers = model.build_model(show_model=False)

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print("Experimental Tag:", tag)

    # load model parameters
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if tag is None else 'params_%s.pkl' % tag
    dump_file = os.path.join(out_path, dump_file)

    print("\n")
    print("Loading model parameters from:", dump_file)
    with open(dump_file, 'rb') as fp:
         params = pickle.load(fp)
    if isinstance(params[0], list):
        # old redundant dump
        for i_layer, layer in enumerate(layers):
            lasagne.layers.set_all_param_values(layer, params[i_layer])
    else:
        # non-redundant dump
        lasagne.layers.set_all_param_values(layers, params)

    # select data
    print("\nLoading data...")
    data = select_data(args.data, args.train_split, args.config, args.seed, test_only=True)

    print("\nCompiling prediction functions...")
    l_view1, l_view2, l_v1latent, l_v2latent = layers

    input_1 = input_2 = [l_view1.input_var, l_view2.input_var]
    compute_v1_latent = theano.function(inputs=input_1,
                                        outputs=lasagne.layers.get_output(l_v1latent, deterministic=True))
    compute_v2_latent = theano.function(inputs=input_2,
                                        outputs=lasagne.layers.get_output(l_v2latent, deterministic=True))

    # iterate test data
    print("Evaluating on test set...")

    # compute output on test set
    eval_set = 'test'
    n_test = args.n_test if args.n_test is not None else data[eval_set].shape[0]
    indices = np.linspace(0, data[eval_set].shape[0] - 1, n_test).astype(np.int)
    X1, X2 = data[eval_set][indices]

    print("Computing embedding space...")
    lv1 = batch_compute2(X1, X2, compute_v1_latent, np.min([100, n_test]), prepare1=model.prepare)
    lv2 = batch_compute2(X1, X2, compute_v2_latent, np.min([100, n_test]), prepare1=model.prepare)
    lv1_cca = lv1
    lv2_cca = lv2

    if args.V2_to_V1:
        lv1_cca, lv2_cca = flip_variables(lv1_cca, lv2_cca)

    # reset n_test
    n_test = lv1_cca.shape[0]

    # show some results
    if args.show:

        # compute pairwise distances
        from scipy.spatial.distance import cdist
        dists = cdist(lv1_cca, lv2_cca, metric="cosine")

        plt.figure("Distance Matrix")
        plt.clf()
        plt.imshow(dists, interpolation='nearest', cmap="magma")
        plt.colorbar()
        plt.axis('off')

        # plt.show(block=True)

        for i in six.moves.range(n_test):
            sorted_idx = np.argsort(dists[i])
            rank = np.nonzero(sorted_idx == i)[0][0]

            # show top 8 retrieval results
            plt.figure('Top 8')
            plt.clf()

            plt.subplot(2, 5, 1)
            plt.imshow(X2[i, 0], cmap='viridis', origin="lower")
            # plt.title("Query", fontsize=22)
            plt.axis('off')

            plt.subplot(2, 5, 2)
            plt.imshow(1.0 - X1[sorted_idx[rank], 0], cmap=plt.cm.gray)
            plt.title("Rank: %d" % rank, fontsize=22)
            plt.axis('off')

            for p in six.moves.range(8):
                plt.subplot(2, 5, p + 3)
                plt.imshow(1.0 - X1[sorted_idx[p], 0], cmap=plt.cm.gray)
                plt.title(p, fontsize=22)
                plt.axis('off')

            plt.show(block=True)

    # clip some dimensions
    max_dim = args.max_dim if args.max_dim is not None else lv1_cca.shape[1]
    lv1_cca = lv1_cca[:, 0:max_dim]
    lv2_cca = lv2_cca[:, 0:max_dim]

    # evaluate retrieval result
    print("V1.shape:", lv1_cca.shape)
    print("V2.shape:", lv2_cca.shape)

    # compute random baseline
    # r_idx = range(lv1_cca.shape[0])
    # np.random.shuffle(r_idx)
    # lv2_cca = lv2_cca[r_idx]

    print("Computing performance measures...")
    mean_rank_te, med_rank_te, dist_te, hit_rates, map = eval_retrieval(lv1_cca, lv2_cca)

    # report hit rates
    recall_at_k = dict()

    print("\nHit Rates:")
    for key in np.sort([*hit_rates.keys()]):
        recall_at_k[key] = float(100 * hit_rates[key]) / n_test
        pk = recall_at_k[key] / key
        print("Top %02d: %.3f (%d) %.3f" % (key, recall_at_k[key], hit_rates[key], pk))

    print("\n")
    print("Median Rank: %.2f (%d)" % (med_rank_te, lv2_cca.shape[0]))
    print("Mean Rank  : %.2f (%d)" % (mean_rank_te, lv2_cca.shape[0]))
    print("Mean Dist  : %.5f " % dist_te)
    print("MAP        : %.3f " % map)

    from scipy.spatial.distance import cdist
    dists = np.diag(cdist(lv1_cca, lv2_cca, metric="cosine"))
    print("Min Dist   : %.5f " % np.min(dists))
    print("Max Dist   : %.5f " % np.max(dists))
    print("Med Dist   : %.5f " % np.median(dists))

    # dump results to yaml file
    if args.dump_results:

        # required for yaml export
        map = float(map)
        med_rank_te = float(med_rank_te)
        keys = ["%d" % k for k in recall_at_k.keys()]
        recall_at_k = dict(zip(keys, recall_at_k.values()))

        results = {"map": map, 'med_rank': med_rank_te, 'recall_at_k': recall_at_k}

        ret_dir = "A2S" if args.V2_to_V1 else "S2A"
        res_file = dump_file.replace("params_", "eval_").replace(".pkl", "_%s.yaml")
        res_file = res_file % ret_dir

        with open(res_file, 'wb') as fp:
            yaml.dump(results, fp, default_flow_style=False)
