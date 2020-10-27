
import pickle
import argparse
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from audio_sheet_retrieval.utils.mutopia_data import NO_AUGMENT, load_split
from audio_sheet_retrieval.retrieval_wrapper import RetrievalWrapper
from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD as ROOT_DIR
from audio_sheet_retrieval.utils.alignment import compute_alignment, estimate_alignment_error

from audio_sheet_retrieval.run_train import compile_tag, select_model

from msmd.midi_parser import extract_spectrogram
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool

sns.set_style('ticks')


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run audio 2 sheet music retrieval service.')
    parser.add_argument('--model', help='select model to train.', default='models/mozart_ccal_cont.py')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--step_sheet', help='number of sheet image snippets.', type=int, default=10)
    parser.add_argument('--step_spec', help='number of spectrogram snippets.', type=int, default=2)
    parser.add_argument('--real_audio', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--align_by', help='select alignment method (baseline, pydtw).', type=str, default='baseline')
    parser.add_argument('--plots', help='show plots.', action='store_true')
    parser.add_argument('--dump_alignment', help='dump results.', action='store_true')
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    args = parser.parse_args()

    # load experiment config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)
    test_augment = NO_AUGMENT.copy()
    test_augment['synths'] = [config["TEST_SYNTH"]]
    test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print("Experimental Tag:", tag)

    # parse arguments
    SHEET_STEP = args.step_sheet
    SPEC_STEP = args.step_spec
    TOL = 25  # tolerance in pixel

    # load retrieval model_name
    model, _ = select_model(args.model)
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if tag is None else 'params_%s.pkl' % tag
    dump_file = os.path.join(out_path, dump_file)

    # get input dimensions
    sheet_win_shape = [1, config['STAFF_HEIGHT'], config['SHEET_CONTEXT']]
    spec_win_shape = [1, config['SPEC_BINS'], config['SPEC_CONTEXT']]

    # initialize embedding network
    prepare_view_1 = model.prepare
    embed_network = RetrievalWrapper(model, spec_shape=spec_win_shape, sheet_shape=sheet_win_shape,
                                     param_file=dump_file, prepare_view_1=prepare_view_1, prepare_view_2=None)



    # select pieces
    split = load_split(args.train_split)
    te_pieces = split["test"]
    pieces = [te_pieces[te_pieces.index("Anonymous__lanative__lanative")]]
    # pieces = te_pieces

    # collect pixel errors for all pieces
    piece_pxl_errors = dict()

    for piece in pieces:

        print("\nTarget Piece: %s" % piece)
        piece_image, piece_specs, piece_o2c_maps, piece_path = prepare_piece_data(ROOT_DIR, piece,
                                                                                  aug_config=test_augment,
                                                                                  require_audio=False)

        # initialize data pool with piece
        piece_pool = AudioScoreRetrievalPool([piece_image], [piece_specs], [piece_o2c_maps], [piece_path],
                                             spec_context=spec_win_shape[2], spec_bins=spec_win_shape[1],
                                             sheet_context=sheet_win_shape[2], staff_height=sheet_win_shape[1],
                                             data_augmentation=test_augment, shuffle=False)

        # compute spectrogram from file
        if args.real_audio:
            audio_file = "/home/matthias/cp/data/sheet_localization/real_music/0_real_audio/%s.flac" % piece
            if not os.path.exists(audio_file):
                continue
            spec = extract_spectrogram(audio_file)
        # use pre-computed spectrogram
        else:
            spec = piece_pool.specs[0][0]

        # get unwrapped sheet image
        sheet = piece_pool.images[0]

        # get coordinates and onsets
        # [0][0] ... piece zero, performance zero
        coords = piece_pool.o2c_maps[0][0][:, 1]
        onsets = piece_pool.o2c_maps[0][0][:, 0]

        # prepare sample points
        n_steps = spec.shape[1] // SPEC_STEP
        o0 = spec_win_shape[2] // 2
        o1 = spec.shape[1] - spec_win_shape[2] // 2
        spec_idxs = np.linspace(o0, o1, n_steps).astype(np.int32)

        n_steps = sheet.shape[1] // SHEET_STEP
        c0 = sheet_win_shape[2] // 2
        c1 = sheet.shape[1] - sheet_win_shape[2] // 2
        sheet_idxs = np.linspace(c0, c1, n_steps).astype(np.int32)

        # slice sheet image
        sheet_slices = np.zeros((len(sheet_idxs), 1, sheet_win_shape[1], sheet_win_shape[2]), dtype=np.float32)
        r0 = sheet.shape[0] // 2 - sheet_win_shape[1] // 2
        r1 = r0 + sheet_win_shape[1]
        for j, x_coord in enumerate(sheet_idxs):
            sheet_slice = sheet[r0:r1, x_coord - c0:x_coord + c0]
            sheet_slices[j, 0] = sheet_slice

        # slice audio
        spec_slices = np.zeros((len(spec_idxs), 1, spec_win_shape[1], spec_win_shape[2]), dtype=np.float32)
        for j, onset in enumerate(spec_idxs):
            spec_slice = spec[:, onset - o0:onset + o0]
            spec_slices[j, 0] = spec_slice

        # compute sheet snippet codes
        img_codes = np.zeros((sheet_slices.shape[0], model.DIM_LATENT), dtype=np.float32)
        for j in range(sheet_slices.shape[0]):
            imges = sheet_slices[j:j + 1]
            img_codes[j] = embed_network.compute_view_1(imges)

        # compute spectrogram snippet codes
        spec_codes = np.zeros((spec_slices.shape[0], model.DIM_LATENT), dtype=np.float32)
        for j in range(spec_slices.shape[0]):
            specs = spec_slices[j:j + 1]
            spec_codes[j] = embed_network.compute_view_2(specs)

        # compute alignment
        a2s_mapping, dtw_res = compute_alignment(img_codes, spec_codes, sheet_idxs, spec_idxs, args.align_by)

        # compute abs value of error
        pxl_errors = estimate_alignment_error(coords, onsets, a2s_mapping)
        abs_pxl_errors = np.abs(pxl_errors)

        # report results for piece
        print("Mean Error:   %.3f" % np.mean(abs_pxl_errors))
        print("Median Error: %.3f" % np.median(abs_pxl_errors))
        print("Max Error:    %.3f" % np.max(abs_pxl_errors))

        piece_pxl_errors[piece] = pxl_errors

        if args.plots:

            # show distance matrix
            plt.figure("Distance Matrix", figsize=(9, 9))
            plt.clf()
            ax = plt.subplot(111)
            plt.imshow(dtw_res['dists'], cmap='magma', interpolation='nearest')
            plt.plot(range(len(spec_codes)), dtw_res['aligned_sheet_idxs'], 'w-', linewidth=3, alpha=0.3)
            plt.xlim([0, dtw_res['dists'].shape[1] - 1])
            plt.ylim([0, dtw_res['dists'].shape[0] - 1])
            plt.ylabel("Sheet (%d)" % len(img_codes), fontsize=20)
            plt.xlabel("Audio (%d)" % len(spec_codes), fontsize=20)
            plt.title("Distance Matrix and DTW Path", fontsize=22)

            # change fontsize of ticks
            ax.tick_params(axis='both', which='major', labelsize=16)

            # show alignment
            plt.figure("Interpolation")
            plt.clf()

            plt.plot(spec_idxs, dtw_res['aligned_sheet_coords'], 'bo', alpha=0.5)
            plt.plot(dtw_res['i_inter'], dtw_res['a2s_alignment'], 'c-', label='alignment')

            for i, o in enumerate(onsets):
                col = 'og' if abs_pxl_errors[i] < TOL else '*m'
                plt.plot(o, coords[i], col, alpha=0.7)

            plt.legend()
            plt.grid(True)
            plt.xlabel("Spectrogram Frame")
            plt.ylabel("Pixel x-Coordinate")
            plt.title("Offline Alignment")

            # visualize result in score
            plt.figure()
            plt.subplots_adjust(left=0.02, right=0.98)

            plt.subplot(2, 1, 1)
            plt.imshow(sheet, cmap=plt.cm.gray)
            plt.colorbar()

            y_coords = np.ones_like(coords) * sheet.shape[0] // 2 - piece_pool.staff_height // 2
            for i, o in enumerate(onsets):
                plt.plot([coords[i], coords[i] + abs_pxl_errors[i]],
                         [y_coords[i], y_coords[i]], '*-', color='b', alpha=0.5)

                col = 'og' if abs_pxl_errors[i] < TOL else 'om'
                plt.plot(coords[i], y_coords[i], col, alpha=1.0)

            plt.xlim([0, sheet.shape[1]-1])
            plt.ylim([sheet.shape[0]-1, 0])

            plt.subplot(2, 1, 2)
            plt.imshow(spec, origin='lower', cmap='viridis')
            for o in onsets:
                plt.plot([o, o], [0, spec.shape[0]], 'w-', alpha=0.5)
            plt.colorbar()
            plt.xlim([0, spec.shape[1] - 1])
            plt.ylim([0, spec.shape[0] - 1])

            plt.show(block=True)

    # dump results for further analysis
    res_file = dump_file.replace("params_", "alignment_res_").replace(".pkl", "_%s.pkl")
    res_file %= args.align_by
    with open(res_file, "wb") as fp:
        pickle.dump(piece_pxl_errors, fp)

    # # dump full alignment for visualizations
    # res_file = os.path.join("res_a2s_align", "alignment_dump_" + model.EXP_NAME + "_" + args.align_by + ".pkl")
    # with open(res_file, "wb") as fp:
    #     data = [spec, sheet, a2s_mapping, dtw_res]
    #     pickle.dump(data, fp)
