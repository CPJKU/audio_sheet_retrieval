
from __future__ import print_function

import os
import yaml
import argparse
import numpy as np

from config.settings import EXP_ROOT
from config.settings import DATA_ROOT_MSMD as ROOT_DIR
from utils.mutopia_data import load_split
from run_train import compile_tag, select_model
from audio_sheet_server import AudioSheetServer
from utils.plotting import BColors
from utils.data_pools import prepare_piece_data, NO_AUGMENT

# init color printer
col = BColors()


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run audio 2 sheet music retrieval service.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--init_audio_db', help='initialize sheet db.', action='store_true')
    parser.add_argument('--full_eval', help='run evaluation on all tracks.', action='store_true')
    parser.add_argument('--running_frames', help='running detection window.', type=int, default=100)
    parser.add_argument('--n_candidates', help='running detection window.', type=int, default=25)
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    parser.add_argument('--dump_results', help='dump results of current run to file.', action='store_true')
    args = parser.parse_args()

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print("Experimental Tag:", tag)

    # load experiment config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)
    test_augment = NO_AUGMENT.copy()
    test_augment['synths'] = [config["TEST_SYNTH"]]
    test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

    # initialize model
    a2s_srv = AudioSheetServer()

    # load tr/va/te split
    split = load_split(args.train_split)
    te_pieces = split["test"]

    # load retrieval model
    model, _ = select_model(args.model)
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if tag is None else 'params_%s.pkl' % tag
    dump_file = os.path.join(out_path, dump_file)
    a2s_srv.initialize_embedding_network(model, param_file=dump_file)

    # initialize sheet music data base
    if args.init_audio_db:
        a2s_srv.initialize_audio_db(pieces=te_pieces, augment=test_augment, keep_snippets=False)
        a2s_srv.save_audio_db_file("audio_db_file.pkl")

    # load sheet music data base
    else:
        a2s_srv.load_audio_db_file("audio_db_file.pkl")

    # run full evaluation
    if args.full_eval:
        print(col.print_colored("\nRunning full evaluation:", col.UNDERLINE))

        ranks = []
        for tp in te_pieces:

            # load piece
            piece_image, _, _ = prepare_piece_data(ROOT_DIR, tp, aug_config=test_augment, require_audio=False)

            # detect piece from spectrogram
            ret_result, ret_votes = a2s_srv.detect_performance(piece_image, top_k=len(te_pieces), n_candidates=args.n_candidates, verbose=False)
            if tp in ret_result:
                rank = ret_result.index(tp) + 1
                ratio = ret_votes[ret_result.index(tp)]
            else:
                rank = len(ret_result)
                ratio = 0.0
            ranks.append(rank)
            color = col.OKBLUE if ranks[-1] == 1 else col.WARNING
            print(col.print_colored("rank: %02d (%.2f) " % (ranks[-1], ratio), color) + tp)

        # report results
        ranks = np.asarray(ranks)
        n_queries = len(ranks)
        for r in xrange(1, n_queries + 1):
            n_correct = np.sum(ranks == r)
            if n_correct > 0:
                print(col.print_colored("%d of %d retrieved performances ranked at position %d." % (n_correct, n_queries, r), col.WARNING))

        # dump retrieval results to file
        if args.dump_results:
            ret_dir = "S2A"
            res_file = dump_file.replace("params_", "retrieval_").replace(".pkl", "_%s.yaml")
            res_file %= ret_dir

            results = [int(r) for r in ranks]
            with open(res_file, 'wb') as fp:
                yaml.dump(results, fp, default_flow_style=False)
