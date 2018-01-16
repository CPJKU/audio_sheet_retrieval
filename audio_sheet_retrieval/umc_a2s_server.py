
from __future__ import print_function

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import EXP_ROOT
from utils.plotting import BColors
from run_train import compile_tag, select_model
from utils.data_pools import SYSTEM_HEIGHT
from audio_sheet_server import AudioSheetServer

from sheet_manager.midi_parser import processor, SAMPLE_RATE, FRAME_SIZE, FPS


# set seaborn style and get colormap
sns.set_style("ticks")
colors = sns.color_palette()

# init color printer
col = BColors()


def spec_gen(Spec):
    """ frame from spectrogram generator """
    for i in xrange(Spec.shape[1]):
        yield Spec[:, i:i+1]


def load_sheets(umc_dir = "/home/matthias/Data/mini_umc/"):
    """
    load unwarpped sheets
    """
    import shutil
    import glob
    import cv2
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from collections import defaultdict

    piece_names = []
    unwrapped_sheets = []
    piece_paths = []

    # get list of all pieces
    piece_dirs = np.sort(glob.glob(os.path.join(umc_dir, '*')))
    n_pieces = len(piece_dirs)

    # iterate pieces
    for i_piece, piece_dir in enumerate(piece_dirs):
        piece_name = piece_dir.split('/')[-1]

        piece_names.append(piece_name)
        piece_paths.append(piece_dir)

        if "Beethoven" not in piece_name:
            continue
        print(col.print_colored("\nProcessing piece %d of %d (%s)" % (i_piece + 1, n_pieces, piece_name), col.OKBLUE))

        # render audio if not there
        trg_audio_path = os.path.join(piece_dir, "audio.flac")
        if not os.path.exists(trg_audio_path):
            from sheet_manager.render_audio import render_audio
            # midi file path
            midi_file_path = os.path.join(piece_dir, "score_ppq.mid")
            audio_path, perf_midi_path = render_audio(midi_file_path, sound_font="grand-piano-YDP-20160804",
                                                      velocity=None, change_tempo=True, tempo_ratio=1.0,
                                                      target_dir=None, quiet=True, audio_fmt=".flac",
                                                      sound_font_root="~/.fluidsynth")
            shutil.copy(audio_path, trg_audio_path)

        # load systems
        page_systems = defaultdict(list)
        page_system_quaters = defaultdict(list)
        system_path = os.path.join(piece_dir, "score_systems.yaml")
        if os.path.exists(system_path):
            print("Systems annotated!")
            with open(system_path, 'rb') as fp:
                yaml_systems = yaml.load(fp)

                for yaml_system in yaml_systems:
                    page_id = yaml_system['page']

                    # convert system coordinates to array
                    system_bbox = np.zeros((4, 2))
                    system_bbox[0] = np.asarray([yaml_system['topLeft']])
                    system_bbox[1] = np.asarray([yaml_system['topRight']])
                    system_bbox[2] = np.asarray([yaml_system['bottomRight']])
                    system_bbox[3] = np.asarray([yaml_system['bottomLeft']])
                    system_bbox = system_bbox[:, ::-1]

                    # keep coordinate if system is not there
                    system_found = False
                    for i, bbox in enumerate(page_systems[page_id]):

                        # compute overlap in y-direction
                        dy_min = min(bbox[2, 0], system_bbox[2, 0]) - max(bbox[1, 0], system_bbox[1, 0])
                        dy_max = max(bbox[2, 0], system_bbox[2, 0]) - min(bbox[1, 0], system_bbox[1, 0])
                        if dy_min >= 0:
                            overlap = dy_min / dy_max
                        else:
                            overlap = 0

                        if overlap == 1:
                            system_found = True
                        elif overlap > 0:
                            system_found = True

                            # merge system coordinates
                            system_bbox[0, 1] = min(bbox[0, 1], system_bbox[0, 1])
                            system_bbox[1, 1] = max(bbox[1, 1], system_bbox[1, 1])
                            system_bbox[2, 1] = max(bbox[1, 1], system_bbox[1, 1])
                            system_bbox[3, 1] = min(bbox[0, 1], system_bbox[0, 1])
                            page_systems[page_id][i] = system_bbox

                            # append quarters covered by system
                            page_system_quaters[page_id][i].append(yaml_system['quarters'])

                    if not system_found:
                        page_systems[page_id].append(system_bbox)
                        page_system_quaters[page_id].append([yaml_system['quarters']])

                # convert coordinates to array
                for page_id in page_systems.keys():
                    page_systems[page_id] = np.asarray(page_systems[page_id], dtype=np.float32)

        else:
            print("No systems annotated!")
            continue

        # load pages
        unwrapped_sheet = np.zeros((SYSTEM_HEIGHT, 0), dtype=np.uint8)
        page_paths = np.sort(glob.glob(os.path.join(piece_dir, "pages/*.png")))
        for i_page, page_path in enumerate(page_paths):
            page_id = i_page + 1
            I = cv2.imread(page_path, 0)

            # resize image
            width = 835
            scale = float(width) / I.shape[1]
            height = int(scale * I.shape[0])
            I = cv2.resize(I, (width, height))

            # re-scale coordinates
            page_systems[page_id] *= scale

            # unwrap sheet
            for system in page_systems[page_id]:
                system = system.astype(np.int)

                r0 = int(np.mean([system[0, 0], system[2, 0]])) - SYSTEM_HEIGHT // 2
                r1 = r0 + SYSTEM_HEIGHT
                c0 = int(system[0, 1])
                c1 = int(system[1, 1])

                unwrapped_sheet = np.hstack((unwrapped_sheet, I[r0:r1, c0:c1].astype(np.uint8)))

            # show sheet image and annotations
            if 0:
                plt.figure("sheet")
                plt.clf()
                ax = plt.subplot(111)

                # plot sheet
                plt.imshow(I, cmap=plt.cm.gray)
                plt.xlim([0, I.shape[1] - 1])
                plt.ylim([I.shape[0] - 1, 0])

                # plot system centers
                system_centers = np.mean(page_systems[page_id][:, [0, 3], 0], axis=1, keepdims=True)
                plt.plot(page_systems[page_id][:, 0, 1], system_centers.flatten(), 'mo')

                # plot systems
                patches = []
                for system_coords in page_systems[page_id]:
                    polygon = Polygon(system_coords[:, ::-1], True)
                    patches.append(polygon)
                p = PatchCollection(patches, color='r', alpha=0.2)
                ax.add_collection(p)

                plt.show(block=True)

        unwrapped_sheets.append(unwrapped_sheet)

    return piece_names, piece_paths, unwrapped_sheets


def load_umc_sheets(data_dir="/home/matthias/Data/umc_mozart", require_performance=False):
    """ load unwarpped sheets """
    import shutil
    import glob
    import cv2

    # initialize omr system
    from omr.omr_app import OpticalMusicRecognizer
    from omr.utils.data import prepare_image
    from lasagne_wrapper.network import SegmentationNetwork

    from omr.models import system_detector, bar_detector

    net = system_detector.build_model()
    system_net = SegmentationNetwork(net, print_architecture=False)
    system_net.load('omr_models/system_params.pkl')

    net = bar_detector.build_model()
    bar_net = SegmentationNetwork(net, print_architecture=False)
    bar_net.load('omr_models/bar_params.pkl')

    piece_names = []
    unwrapped_sheets = []
    piece_paths = []

    # get list of all pieces
    piece_dirs = np.sort(glob.glob(os.path.join(data_dir, '*')))
    n_pieces = len(piece_dirs)

    # iterate pieces
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

        unwrapped_sheet = np.zeros((SYSTEM_HEIGHT, 0), dtype=np.uint8)
        system_problem = False
        for i_page, page_path in enumerate(page_paths):
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

                r0 = int(np.mean([system[0, 0], system[2, 0]])) - SYSTEM_HEIGHT // 2
                r1 = r0 + SYSTEM_HEIGHT
                c0 = int(system[0, 1])
                c1 = int(system[1, 1])

                # fix row slice coordinates
                r0 = max(0, r0)
                r1 = min(r1, I.shape[0])
                r0 = max(r0, r1 - SYSTEM_HEIGHT)

                staff_img = I[r0:r1, c0:c1].astype(np.uint8)

                if staff_img.shape[0] < SYSTEM_HEIGHT:
                    to_pad = SYSTEM_HEIGHT - staff_img.shape[0]
                    if to_pad > (0.1 * SYSTEM_HEIGHT):
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

    # define test pieces
    te_pieces, piece_paths, unwrapped_sheets = load_umc_sheets(args.data_dir, require_performance=True)
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
                audio_file = os.path.join(piece_paths[i], "01_performance.wav")
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
            color = col.OKBLUE if ranks[-1] == 1 else col.WARNING
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
