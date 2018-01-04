
from __future__ import print_function

import os
import sys
import yaml
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import cdist

from config.settings import EXP_ROOT
# todo: fix this to augmented directory
from config.settings import DATA_ROOT_MSMD as ROOT_DIR
from utils.data_pools import SPEC_CONTEXT, SPEC_BINS, NO_AUGMENT
from utils.mutopia_data import load_split
from utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool
from utils.plotting import BColors
from run_train import compile_tag, select_model
from retrieval_wrapper import RetrievalWrapper

from sheet_manager.midi_parser import processor, SAMPLE_RATE, FRAME_SIZE, FPS

# todo: remove this
NO_AUGMENT['synths'] = ['FluidR3_GM']


# set seaborn style and get colormap
sns.set_style("ticks")
colors = sns.color_palette()

# init color printer
col = BColors()


def spec_gen(Spec):
    """ frame from spectrogram generator """
    for i in xrange(Spec.shape[1]):
        yield Spec[:, i:i+1]


def online_frame_generator():
    """ generate signal frames from mic """
    from madmom.audio.signal import Stream
    hop_size = int(SAMPLE_RATE / FPS)
    stream = Stream(sample_rate=SAMPLE_RATE, num_channels=1, frame_size=FRAME_SIZE,
                    hop_size=hop_size, queue_size=1)

    for frame in stream:
        yield frame


def resize_image(img, rsz=1.0):
    import cv2
    new_shape = (int(img.shape[1] * rsz), int(img.shape[0] * rsz))
    return cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)


class AudioSheetServer(object):
    """ Audio to Sheet Music Retrieval Server """

    def __init__(self):
        """ constructor """

        self.sheet_snippet_codes = None
        self.sheet_snippet_ids = None
        self.id_to_piece = None
        self.sheet_snippets = None

        self.embed_network = None
        self.snippet_shape = None

        self.spec_shape = (SPEC_BINS, SPEC_CONTEXT)

    def run(self, spec=None, top_k=5, n_candidates=5, running_frames=None, gui=True, target_piece=None):
        """
        run sheet retrieval service
            top_k: number of pieces visualized in histogram
            n_candidates: number of condidates taken into account for each frame
            running_frames: number of frame history for histogram computation
        """
        print("Running server ...")

        running_spec = np.zeros((self.spec_shape[0], self.spec_shape[1]), dtype=np.float32)

        if spec is None:
            spectrogram_generator = self._mic_spec_gen()
        else:
            spectrogram_generator = spec_gen(spec)

        try:

            all_piece_ids = np.zeros(0, dtype=np.int)
            sorted_count_idxs = None
            frame_times = np.zeros(10)
            for i_frame, Frame in enumerate(spectrogram_generator):

                # estimate fps
                start = time.time()

                # keep sliding window spectrogram
                running_spec = np.hstack((running_spec[:, 1::], Frame))

                # compute music probability
                m_prob = self._detect_music(running_spec, spec)

                # check if music is played
                M_THRESH = 0.5
                if m_prob > M_THRESH and i_frame >= running_spec.shape[1]:

                    # compute spec code
                    spec_code = self.embed_network.compute_view_2(running_spec[np.newaxis, np.newaxis, :, :])

                    # retrive pice ids for current spectrogram
                    piece_ids, snippet_ids = self._retrieve_sheet_snippet_ids(spec_code, n_candidates=n_candidates)

                    # keep piece ids
                    all_piece_ids = np.concatenate((all_piece_ids, piece_ids))
                    first_idx = running_frames * n_candidates
                    if running_frames is not None and all_piece_ids.shape[0] > first_idx:
                        all_piece_ids = all_piece_ids[-first_idx:]

                    # count voting for each piece
                    unique, counts = np.unique(all_piece_ids, return_counts=True)

                    # normalize counts to probability
                    counts = counts.astype(np.float) / np.sum(counts)

                    # get top k pieces
                    sorted_count_idxs = np.argsort(counts)[::-1][:top_k]

                # show histogram
                if gui:
                    plt.figure("SheetMusicRetrievalServer", figsize=(10, 10))
                    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 3], width_ratios=[1, 1])
                    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=0.6)
                    plt.clf()

                    plt.subplot(gs[0])
                    plt.title("Incoming Audio %d" % i_frame, fontsize=20)
                    plt.imshow(running_spec, cmap='viridis', origin="lower")
                    plt.grid('off')
                    plt.axis('off')

                    plt.subplot(gs[1])
                    plt.title("Music Probability", fontsize=20)
                    plt.bar([0.15], [m_prob], width=0.2, color=colors[0], alpha=0.8)
                    plt.plot([0.1, 0.52], [M_THRESH, M_THRESH], '-', color=colors[2], linewidth=5.0, alpha=0.5)
                    plt.text(0.52, M_THRESH + 0.01, 'Music', color=colors[2], va="bottom", ha="right", fontsize=18)
                    plt.xlim([-0.1, 0.52])
                    plt.ylim([0, 1.05])
                    plt.axis('off')

                    plt.subplot(gs[2:4])
                    plt.title("Piece Retrieval Ranking", fontsize=20)
                    plt.ylabel("Piece Probability", fontsize=18)
                    plt.xlim([-0.5, top_k])
                    plt.ylim([0.0, 1.0])
                    if sorted_count_idxs is not None:
                        x_coords = range(len(sorted_count_idxs))
                        plt.bar(x_coords, counts[sorted_count_idxs], width=0.5, color=colors[0])
                        labels = [self.id_to_piece[unique[idx]] for idx in sorted_count_idxs]

                        # highlight target piece
                        if target_piece and target_piece in labels:
                            target_idx = labels.index(target_piece)
                            ticks = plt.gca().xaxis.get_major_ticks()
                            ticks[target_idx].label.set_fontweight("bold")

                            plt.bar(x_coords[target_idx], counts[sorted_count_idxs][target_idx], width=0.5, color=colors[2])

                        plt.xticks(x_coords, labels, rotation=15)

                        # snippet visualization
                        snippet_imgs = []
                        for i_snippet, id in enumerate(snippet_ids):
                            snippet = self.sheet_snippets[id]
                            if self.id_to_piece[piece_ids[i_snippet]] != target_piece:
                                snippet = 255 - snippet
                            snippet = np.pad(snippet, ((2, 2), (2, 2)), mode="constant", constant_values=125)
                            snippet_imgs.append(snippet)
                        snippet_imgs = np.vstack((np.hstack(snippet_imgs[0:8]),
                                                  np.hstack(snippet_imgs[8:16]),
                                                  np.hstack(snippet_imgs[16:24])))
                        plt.subplot(gs[4:6])
                        plt.imshow(snippet_imgs, cmap=plt.cm.gray)
                        plt.axis("off")
                        plt.title("Top-k Retrieved Snippets", fontsize=20)

                    plt.draw()
                    plt.pause(0.01)
                    plt.savefig("figs/%05d.png" % i_frame)

                # estimate fps
                stop = time.time()
                frame_times[1:] = frame_times[0:-1]
                frame_times[0] = stop - start
                fps = 1.0 / np.mean(frame_times)
                print("Server is running at %.2f fps." % fps, end='\r')
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nStopping server ...")

    def detect_piece(self, spectrogram, top_k=1, n_candidates=1, verbose=False):
        """ detect piece from audio """

        all_piece_ids = np.zeros(0, dtype=np.int)

        n_samples = 100
        start_indices = np.linspace(start=0, stop=spectrogram.shape[1]-self.spec_shape[1], num=n_samples)
        start_indices = start_indices.astype(np.int)

        for idx in start_indices:
            spec_snippet = spectrogram[:, idx:idx+self.spec_shape[1]]
            spec_snippet = spec_snippet[np.newaxis, np.newaxis, :, :]

            # compute spec code
            spec_code = self.embed_network.compute_view_2(spec_snippet)

            # retrive pice ids for current spectrogram
            piece_ids, snippet_ids = self._retrieve_sheet_snippet_ids(spec_code, n_candidates=n_candidates)

            # keep piece ids
            all_piece_ids = np.concatenate((all_piece_ids, piece_ids))

        # count voting for each piece
        unique, counts = np.unique(all_piece_ids, return_counts=True)

        # return top k pieces
        sorted_count_idxs = np.argsort(counts)[::-1][:top_k]

        # report
        if verbose:
            print(col.print_colored("\nRetrieval Ranking:", color=col.UNDERLINE))
            for idx in sorted_count_idxs:
                print("pid: %03d (%03d): %s" % (unique[idx], counts[idx], self.id_to_piece[unique[idx]]))
            print("")

        ret_result = [self.id_to_piece[unique[idx]] for idx in sorted_count_idxs]
        ret_votes = [counts[idx] for idx in sorted_count_idxs]
        ret_votes = np.asarray(ret_votes, dtype=np.float) / np.sum(ret_votes)

        return ret_result, ret_votes

    def initialize_embedding_network(self, model, param_file):
        """ load cross modality retrieval model """
        prepare_view_1 = model.prepare
        self.embed_network = RetrievalWrapper(model, param_file, prepare_view_1=prepare_view_1, prepare_view_2=None)
        self.snippet_shape = self.embed_network.shape_view1[1:]

    def initialize_sheet_db(self, pieces, keep_snippets=True):
        """ load sheet music data base """
        print("Initializing sheet music db ...")

        self.id_to_piece = dict()
        self.sheet_snippet_ids = np.zeros(0, dtype=np.int)
        self.sheet_snippet_codes = np.zeros((0, self.embed_network.code_dim), dtype=np.float32)
        self.sheet_snippets = np.zeros((0, self.snippet_shape[0] // 2, self.snippet_shape[1] // 2), dtype=np.uint8)

        # initialize retrieval pool
        for pice_idx, piece in enumerate(pieces):
            print(" (%03d / %03d) %s" % (pice_idx + 1, len(pieces), piece))

            # load piece
            self.id_to_piece[pice_idx] = piece
            piece_image, piece_specs, piece_o2c_maps = prepare_piece_data(ROOT_DIR, piece, require_audio=False)

            # initialize data pool with piece
            piece_pool = AudioScoreRetrievalPool([piece_image], [piece_specs], [piece_o2c_maps],
                                                 data_augmentation=NO_AUGMENT, shuffle=False)

            # embed sheet snippets of piece
            for j in xrange(piece_pool.shape[0]):

                # get image snippet
                image, _ = piece_pool[j:j+1]

                # compute sheet snippet code
                code = self.embed_network.compute_view_1(image)

                # keep code
                self.sheet_snippet_codes = np.concatenate((self.sheet_snippet_codes, code))

                # keep sheet snippets
                if keep_snippets:
                    snippet = resize_image(image[0, 0], rsz=0.5).astype(np.uint8)[np.newaxis]
                    self.sheet_snippets = np.concatenate((self.sheet_snippets, snippet))

            # save id of piece
            pice_ids = np.ones(piece_pool.shape[0], dtype=np.int) * pice_idx
            self.sheet_snippet_ids = np.concatenate((self.sheet_snippet_ids, pice_ids))

        print("%s sheet snippet codes of %d pieces collected" % (self.sheet_snippet_codes.shape[0], len(pieces)))

    def load_sheet_db_file(self, sheet_db_path):
        """ load sheet codes """
        print("Loading sheet db codes ...")
        with open(sheet_db_path, 'rb') as fp:
            data = pickle.load(fp)
            self.sheet_snippet_codes, self.sheet_snippet_ids, self.id_to_piece, self.sheet_snippets = data

    def save_sheet_db_file(self, sheet_db_path):
        """ preserve sheet codes """
        print("Dumping sheet db codes ...")
        with open(sheet_db_path, 'wb') as fp:
            data = [self.sheet_snippet_codes, self.sheet_snippet_ids, self.id_to_piece, self.sheet_snippets]
            pickle.dump(data, fp)

    def _detect_music(self, running_spec, spec):
        """ detect if music is played """
        music_prob = running_spec.sum(axis=0).mean()
        music_prob /= (spec.sum(axis=0).max() * 0.15)
        return np.clip(music_prob, 0.0, 1.0)

    def _retrieve_sheet_snippet_ids(self, spectrogram_code, n_candidates=1):
        """ retrieve k most similar sheet music snippets """

        # compute distance
        dists = cdist(self.sheet_snippet_codes, spectrogram_code, metric="cosine").flatten()

        # sort indices by distance
        sorted_idx = np.argsort(dists)[:n_candidates]

        # plt.figure()
        # plt.subplot(4, 1, 1)
        # plt.plot(dists)
        # plt.subplot(4, 1, 2)
        # plt.plot(self.sheet_snippet_ids)
        # plt.subplot(4, 1, 3)
        # plt.plot(dists[np.argsort(dists)])
        # plt.subplot(4, 1, 4)
        # plt.plot(self.sheet_snippet_ids[np.argsort(dists)])
        # plt.show(block=True)

        # return piece ids
        return self.sheet_snippet_ids[sorted_idx], sorted_idx


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run audio 2 sheet music retrieval service.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--init_db', help='initialize sheed db.', action='store_true')
    parser.add_argument('--full_eval', help='run evaluation on all tracks.', action='store_true')
    parser.add_argument('--real_audio', help='use real audio recordings.', action='store_true')
    parser.add_argument('--running_frames', help='running detection window.', type=int, default=100)
    parser.add_argument('--n_candidates', help='running detection window.', type=int, default=25)
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    args = parser.parse_args()

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print("Experimental Tag:", tag)

    # load experiment config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)
    synth = config["TEST_SYNTH"]

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
    if args.init_db:
        a2s_srv.initialize_sheet_db(pieces=te_pieces, keep_snippets=True)
        a2s_srv.save_sheet_db_file("sheet_db_file.pkl")

    # load sheet music data base
    else:
        a2s_srv.load_sheet_db_file("sheet_db_file.pkl")

    # run full evaluation
    if args.full_eval:
        print(col.print_colored("\nRunning full evaluation:", col.UNDERLINE))

        ranks = []
        for tp in te_pieces:

            # compute spectrogram from file
            if args.real_audio:
                audio_file = os.path.join(ROOT_DIR, "0_real_audio/%s.flac" % tp)
            else:
                audio_file = os.path.join(ROOT_DIR, "%s/performances/%s_tempo-1000_%s/%s_tempo-1000_%s.flac")
                audio_file %= (tp, tp, synth, tp, synth)

            if not os.path.exists(audio_file):
                continue

            spec = processor.process(audio_file).T

            # detect piece from spectrogram
            ret_result, ret_votes = a2s_srv.detect_piece(spec, top_k=len(te_pieces), n_candidates=args.n_candidates, verbose=False)
            if tp in ret_result:
                rank = ret_result.index(tp) + 1
                ratio = ret_votes[ret_result.index(tp)]
            else:
                rank = len(ret_result)
                ratio = 0.0
            ranks.append(rank)
            color = col.OKBLUE if ranks[-1] == 1 else col.WARNING
            print(col.print_colored("rank: %02d (%.2f) " % (ranks[-1], ratio), color) + tp)

        ranks = np.asarray(ranks)
        n_queries = len(ranks)
        for r in xrange(1, n_queries + 1):
            n_correct = np.sum(ranks == r)
            if n_correct > 0:
                print(col.print_colored("%d of %d retrieved pieces ranked at position %d." % (n_correct, n_queries, r), col.WARNING))

    else:

        # compute spectrogram from audio file
        tp = "BachJS__BWV830__BWV-830-2"
        if args.real_audio:
            audio_file = "/home/matthias/cp/data/sheet_localization/real_music/0_retrieval_samples/%s.flac" % tp
        else:
            audio_file = os.path.join(ROOT_DIR, "%s/performances/%s_tempo-1000_%s/%s_tempo-1000_%s.flac")
            audio_file %= (tp, tp, synth, tp, synth)

        spec = processor.process(audio_file).T

        print(col.print_colored("\nQuery Audio: %s" % os.path.basename(audio_file), color=col.OKBLUE))

        # detect piece from spectrogram
        a2s_srv.detect_piece(spec, top_k=7, n_candidates=args.n_candidates, verbose=True)

        # start service
        a2s_srv.run(spec, top_k=7, n_candidates=args.n_candidates, running_frames=args.running_frames, target_piece=tp)
