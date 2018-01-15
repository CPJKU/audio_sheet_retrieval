
from __future__ import print_function

import os
import yaml
import argparse
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

from config.settings import EXP_ROOT
from utils.plotting import BColors
from run_train import compile_tag, select_model
from audio_sheet_server import AudioSheetServer
from umc_a2s_server import load_umc_sheets

from sheet_manager.midi_parser import processor, SAMPLE_RATE, FRAME_SIZE, FPS


# set seaborn style and get colormap
sns.set_style("ticks")
colors = sns.color_palette()

# init color printer
col = BColors()


def load_specs(piece_paths, audio_file):
    """ Compute spectrograms given piece paths """

    spectrograms = []

    for piece_path in piece_paths:
        audio_path = os.path.join(piece_path, audio_file)
        spec = processor.process(audio_path).T
        spectrograms.append(spec)

    return spectrograms

if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run audio 2 sheet music retrieval service.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--init_audio_db', help='initialize sheet db.', action='store_true')
    parser.add_argument('--full_eval', help='run evaluation on all tracks.', action='store_true')
    parser.add_argument('--real_perf', help='use real audio recordings.', action='store_true')
    parser.add_argument('--n_candidates', help='running detection window.', type=int, default=25)
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    parser.add_argument('--dump_results', help='dump results of current run to file.', action='store_true')
    parser.add_argument('--data_dir', help='path to evaluation data.', type=str, default=None)
    args = parser.parse_args()

    # define test pieces
    te_pieces, piece_paths, unwrapped_sheets = load_umc_sheets(args.data_dir, require_performance=True)
    dset = os.path.basename(args.data_dir)

    # load corresponding performance spectrograms
    print("Loading spectrograms ...")
    audio_file = "01_performance.wav" if args.real_perf else "score_ppq.flac"
    spectrograms = load_specs(piece_paths, audio_file=audio_file)

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print("Experimental Tag:", tag)

    # initialize model
    a2s_srv = AudioSheetServer()

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
        a2s_srv.initialize_audio_db_from_specs(pieces=te_pieces, spectrograms=spectrograms,
                                               keep_snippets=False)
        a2s_srv.save_audio_db_file("umc_audio_db_file.pkl")

    # load sheet music data base
    else:
        a2s_srv.load_audio_db_file("umc_audio_db_file.pkl")

    # run full evaluation
    if args.full_eval:
        print(col.print_colored("\nRunning full evaluation:", col.UNDERLINE))

        ranks = []
        for i, tp in enumerate(te_pieces):

            # get sheet music of current piece
            sheet = unwrapped_sheets[i]

            # detect piece from spectrogram
            ret_result, ret_votes = a2s_srv.detect_performance(sheet, top_k=len(te_pieces),
                                                               n_candidates=args.n_candidates, verbose=False)
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
                print(col.print_colored("%d of %d retrieved scores ranked at position %d." % (n_correct, n_queries, r), col.WARNING))

        # dump retrieval results to file
        if args.dump_results:
            ret_dir = "S2A"
            if args.real_perf:
                ret_dir += "_real"
            res_file = dump_file.replace("params_", "umc_retrieval_").replace(".pkl", "_%s_%s.yaml")
            res_file %= (dset, ret_dir)

            results = [int(r) for r in ranks]
            with open(res_file, 'wb') as fp:
                yaml.dump(results, fp, default_flow_style=False)
