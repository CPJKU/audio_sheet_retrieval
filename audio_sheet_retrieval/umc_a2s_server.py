
from __future__ import print_function

import os
import glob
import yaml
import argparse
import numpy as np
import seaborn as sns

from config.settings import EXP_ROOT
from utils.plotting import BColors
from run_train import compile_tag, select_model
from audio_sheet_server import AudioSheetServer

from msmd.midi_parser import processor, SAMPLE_RATE, FRAME_SIZE, FPS


# set seaborn style and get colormap
sns.set_style("ticks")
colors = sns.color_palette()

# init color printer
col = BColors()


def get_performance_audio_path(piece_path, file_pattern):
    """ pass """
    audio_path = os.path.join(piece_path, file_pattern + "*")
    audio_path = glob.glob(audio_path)[0]
    return audio_path


def load_specs(piece_paths, audio_file):
    """ Compute spectrograms given piece paths """

    spectrograms = []

    for piece_path in piece_paths:
        audio_path = get_performance_audio_path(piece_path, audio_file)
        spec = processor.process(audio_path).T
        spectrograms.append(spec)

    return spectrograms


def spec_gen(Spec):
    """ frame from spectrogram generator """
    for i in xrange(Spec.shape[1]):
        yield Spec[:, i:i+1]


def load_umc_sheets(data_dir="/home/matthias/Data/umc_mozart", require_performance=False
                    staff_height=None):
    """ load unwarpped sheets """
    import glob
    import cv2

    # initialize omr system
    from omr.omr_app import OpticalMusicRecognizer
    from omr.utils.data import prepare_image
    from lasagne_wrapper.network import SegmentationNetwork

    from omr.models import system_detector, bar_detector

    net = system_detector.build_model()
    system_net = SegmentationNetwork(net, print_architecture=False)
    system_net.load('sheet_utils/omr_models/system_params.pkl')

    net = bar_detector.build_model()
    bar_net = SegmentationNetwork(net, print_architecture=False)
    bar_net.load('sheet_utils/omr_models/bar_params.pkl')

    piece_names = []
    unwrapped_sheets = []
    piece_paths = []

    # get list of all pieces
    piece_dirs = np.sort(glob.glob(os.path.join(data_dir, '*')))
    n_pieces = len(piece_dirs)

    # iterate pieces
    kept_pages = 0
    for i_piece, piece_dir in enumerate(piece_dirs):
        piece_name = piece_dir.split('/')[-1]

        # if "214_" not in piece_name:
        #     continue

        print(col.print_colored("Processing piece %d of %d (%s)" % (i_piece + 1, n_pieces, piece_name), col.OKBLUE))

        # check if there is a performance
        if require_performance and len(glob.glob(os.path.join(piece_dir, "*performance*"))) == 0:
            print("No performance found!")
            continue

        # load pages
        page_paths = np.sort(glob.glob(os.path.join(piece_dir, "sheet/*.png")))
        if len(page_paths) == 0:
            print("No sheet available!!!")
            continue

        unwrapped_sheet = np.zeros((staff_height, 0), dtype=np.uint8)
        system_problem = False
        for i_page, page_path in enumerate(page_paths):
            kept_pages += 1

            # load sheet image
            I = cv2.imread(page_path, 0)

            # load system coordinates
            # page_id = i_page + 1
            # page_systems = np.load(os.path.join(piece_dir, "coords", "systems_%02d.npy" % (i_page + 1)))

            # detect systems
            I_prep = prepare_image(I)
            omr = OpticalMusicRecognizer(note_detector=None, system_detector=system_net, bar_detector=bar_net)

            try:
                page_systems = omr.detect_systems(I_prep, verbose=False)
            except:
                print("Problem in system detection!!!")
                system_problem = True
                continue

            # plt.figure("System Localization")
            # plt.clf()
            # plt.imshow(I, cmap=plt.cm.gray)
            # plt.xlim([0, I.shape[1] - 1])
            # plt.ylim([I.shape[0] - 1, 0])

            # for system in page_systems:
            #     plt.plot(system[:, 1], system[:, 0], 'mo', alpha=0.5)
            # plt.show(block=True)

            # unwrap sheet
            for system in page_systems:

                r0 = int(np.mean([system[0, 0], system[2, 0]])) - staff_height // 2
                r1 = r0 + staff_height
                c0 = int(system[0, 1])
                c1 = int(system[1, 1])

                # fix row slice coordinates
                r0 = max(0, r0)
                r1 = min(r1, I.shape[0])
                r0 = max(r0, r1 - staff_height)

                staff_img = I[r0:r1, c0:c1].astype(np.uint8)

                if staff_img.shape[0] < staff_height:
                    to_pad = staff_height - staff_img.shape[0]
                    if to_pad > (0.1 * staff_height):
                        print("Problem in system padding!!!")
                        continue
                    staff_img = np.pad(staff_img, ((0, to_pad), (0, 0)), mode="edge")

                unwrapped_sheet = np.hstack((unwrapped_sheet, staff_img))

            # plt.figure("Unwrapped")
            # plt.imshow(unwrapped_sheet)
            # plt.show(block=True)

        if not system_problem:
            piece_names.append(piece_name)
            piece_paths.append(piece_dir)
            unwrapped_sheets.append(unwrapped_sheet)

    print("%d pieces covering %d pages of sheet music." % (len(piece_names), kept_pages))

    return piece_names, piece_paths, unwrapped_sheets


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run audio 2 sheet music retrieval service.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--init_sheet_db', help='initialize sheet db.', action='store_true')
    parser.add_argument('--full_eval', help='run evaluation on all tracks.', action='store_true')
    parser.add_argument('--real_perf', help='use real audio recordings.', action='store_true')
    parser.add_argument('--n_candidates', help='running detection window.', type=int, default=25)
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    parser.add_argument('--dump_results', help='dump results of current run to file.', action='store_true')
    parser.add_argument('--data_dir', help='path to evaluation data.', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)

    # define test pieces
    te_pieces, piece_paths, unwrapped_sheets = load_umc_sheets(args.data_dir, require_performance=True,
                                                               staff_height=config['STAFF_HEIGHT'])
    dset = os.path.basename(args.data_dir)

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
    if args.init_sheet_db:
        a2s_srv.initialize_sheet_db_from_imges(pieces=te_pieces, scores=unwrapped_sheets,
                                               keep_snippets=False)
        a2s_srv.save_sheet_db_file("umc_sheet_db_file.pkl")

    # load sheet music data base
    else:
        a2s_srv.load_sheet_db_file("umc_sheet_db_file.pkl")

    # run full evaluation
    if args.full_eval:
        print(col.print_colored("\nRunning full evaluation:", col.UNDERLINE))

        ranks = []
        for i, tp in enumerate(te_pieces):

            # compute spectrogram from file
            if args.real_perf:
                audio_file = get_performance_audio_path(piece_paths[i], "01_performance")
            else:
                audio_file = os.path.join(piece_paths[i], "score_ppq.flac")

            if not os.path.exists(audio_file):
                continue

            # compute spectrogram
            spec = processor.process(audio_file).T

            # detect piece from spectrogram
            ret_result, ret_votes = a2s_srv.detect_score(spec, top_k=len(te_pieces), n_candidates=args.n_candidates,
                                                         verbose=False)
            if tp in ret_result:
                rank = ret_result.index(tp) + 1
                ratio = ret_votes[ret_result.index(tp)]
            else:
                rank = len(ret_result)
                ratio = 0.0
            ranks.append(rank)
            if ranks[-1] == 1:
                color = col.OKGREEN
            elif ranks[-1] <= 5:
                color = col.OKBLUE
            else:
                color = col.WARNING

            print(col.print_colored("rank: %02d (%.2f) " % (ranks[-1], ratio), color) + tp)

        # report results
        ranks = np.asarray(ranks)
        n_queries = len(ranks)
        for r in xrange(1, n_queries + 1):
            n_correct = np.sum(ranks == r)
            if n_correct > 0:
                print(col.print_colored("%d of %d retrieved scores ranked at position %d." % (n_correct, n_queries, r),
                                        col.WARNING))

        # dump retrieval results to file
        if args.dump_results:
            ret_dir = "A2S"
            if args.real_perf:
                ret_dir += "_real"
            res_file = dump_file.replace("params_", "umc_retrieval_").replace(".pkl", "_%s_%s.yaml")
            res_file %= (dset, ret_dir)

            results = [int(r) for r in ranks]
            with open(res_file, 'wb') as fp:
                yaml.dump(results, fp, default_flow_style=False)
