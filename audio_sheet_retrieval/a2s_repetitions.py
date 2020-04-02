from __future__ import print_function

import os
import sys
import yaml
import json
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import cdist

from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD as ROOT_DIR
from audio_sheet_retrieval.utils.mutopia_data import load_split
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool
from audio_sheet_retrieval.utils.plotting import BColors
from audio_sheet_retrieval.run_train import compile_tag, select_model
from audio_sheet_retrieval.retrieval_wrapper import RetrievalWrapper
from audio_sheet_retrieval.utils.data_pools import NO_AUGMENT
from audio_sheet_retrieval.audio_sheet_server import AudioSheetServer

from msmd.midi_parser import extract_spectrogram
from msmd.data_model.piece import Piece

# init color printer
col = BColors()


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run audio 2 sheet music retrieval service.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--audio_path', help='path to the audio file.')
    parser.add_argument('--piece', help='name of the piece.')
    parser.add_argument('--init_sheet_db', help='initialize sheet db.', action='store_true')
    parser.add_argument('--full_eval', help='run evaluation on all tracks.', action='store_true')
    parser.add_argument('--real_audio', help='use real audio recordings.', action='store_true')
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
    synth = config["TEST_SYNTH"]

    # initialize model
    a2s_srv = AudioSheetServer(spec_shape=(1, config['SPEC_BINS'], config['SPEC_CONTEXT']),
                               sheet_shape=(1, config['STAFF_HEIGHT'], config['SHEET_CONTEXT']))

    # piece name for loading the scores files
    # piece_name = 'BachJS__BWVAnh113__anna-magdalena-03'
    # piece_name = 'BachJS__BWVAnh120__BWV-120'
    # piece_name = 'BachJS__BWVAnh131__air'
    # piece_name = 'BachJS__BWVAnh691__BWV-691'
    # piece_name = [piece_name]
    piece_name = [args.piece]


    # load retrieval model
    model, _ = select_model(args.model)
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if tag is None else 'params_%s.pkl' % tag
    dump_file = os.path.join(out_path, dump_file)
    a2s_srv.initialize_embedding_network(model, param_file=dump_file)

    # initialize sheet music data base
    # for our case, the data base of codes consists of only the target piece
    a2s_srv.initialize_sheet_db(pieces=piece_name, keep_snippets=False)

    # Unrolling the bounding boxes
    # loading the json files
    bb_tag = '/repetitions/' + args.piece + '_barboxes.json'
    with open(bb_tag) as fp:
        bar_boxes = json.load(fp)

    # piece loading
    piece = Piece(root=ROOT_DIR, name=args.piece)
    score = piece.load_score(piece.available_scores[0])

    # get mungos
    mungos = score.load_mungos()
    mdict = {m.objid: m for m in mungos}
    mungos_per_page = score.load_mungos(by_page=True)

    # load images
    images = score.load_images()

    # stack sheet images
    image, page_mungos, mdict = stack_images(images, mungos_per_page, mdict)

    # get only system mungos for unwrapping
    system_mungos = [c for c in page_mungos if c.clsname == 'staff']
    system_mungos = sorted(system_mungos, key=lambda m: m.top)

    # unwrap sheet images
    # un_wrapped_image, un_wrapped_coords = unwrap_sheet_image(image, system_mungos, mdict)

    # get rois from page systems
    rois = systems_to_rois(system_mungos, window_top, window_bottom)

    width = image.shape[1] * rois.shape[0]
    window = rois[0, 3, 0] - rois[0, 0, 0]

    un_wrapped_coords = dict()
    un_wrapped_image = np.zeros((window, width), dtype=np.uint8)

    # make single staff image
    x_offset = 0
    img_start = 0
    for j, sys_mungo in enumerate(system_mungos):

        # get current roi
        r = rois[j]

        # fix out of image errors
        pad_top = 0
        pad_bottom = 0
        if r[0, 0] < 0:
            pad_top = np.abs(r[0, 0])
            r[0, 0] = 0

        if r[3, 0] >= image.shape[0]:
            pad_bottom = r[3, 0] - image.shape[0]

        # get system image
        system_image = image[r[0, 0]:r[3, 0], r[0, 1]:r[1, 1]]

        # pad missing rows and fix coordinates
        system_image = np.pad(system_image, ((pad_top, pad_bottom), (0, 0)), mode='edge')

        img_end = img_start + system_image.shape[1]
        un_wrapped_image[:, img_start:img_end] = system_image

        # get noteheads of current staff
        staff_noteheads = [mdict[i] for i in sys_mungo.inlinks if mdict[i].clsname == 'notehead-full']

        # compute unwraped coordinates
        for n in staff_noteheads:
            n.x -= r[0, 0]
            n.y += x_offset - r[0, 1]
            un_wrapped_coords[n.objid] = n

        x_offset += (r[1, 1] - r[0, 1])
        img_start = img_end

    # get relevant part of unwrapped image
    un_wrapped_image = un_wrapped_image[:, :img_end]

    # compute spectrogram from audio file
    audio_file = args.audio_path
    spec = extract_spectrogram(audio_file)

    print(col.print_colored("\nQuery Audio: %s" % os.path.basename(audio_file), color=col.OKBLUE))

    # computing the embeddings for the spectrogram
    n_samples = 100
    start_indices = np.linspace(start=0, stop=spec.shape[1] - a2s_srv.spec_shape[2], num=n_samples)
    start_indices = start_indices.astype(np.int)

    # collect spectrogram excerpts
    spec_excerpts = np.zeros((len(start_indices), 1, a2s_srv.spec_shape[1], a2s_srv.spec_shape[2]), dtype=np.float32)
    for i, idx in enumerate(start_indices):
        spec_excerpts[i, 0] = spec[:, idx:idx + a2s_srv.spec_shape[2]]

    # compute spec codes
    spec_codes = a2s_srv.embed_network.compute_view_2(spec_excerpts)

    # computing the distances from audio to score
    euc_dists = cdist(a2s_srv.sheet_snippet_codes, spec_codes, metric='euclidean')

    # plotting the distance matrix
    fig, ax = plt.subplots()
    ax.imshow(euc_dists, cmap='magma', interpolation='nearest', aspect='auto')
    fig.savefig("repetitions/dist_matrix_" + os.path.basename(audio_file) + ".png")
    # plt.show()
