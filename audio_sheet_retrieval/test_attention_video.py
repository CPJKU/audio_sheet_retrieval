"""
Visualize Attention model for a single piece from the test data as a video.

Example Call:
    python test_attention_video.py --model models/mutopia_ccal_cont_rsz_dense_att.py \
     --data mutopia --train_split ../../msmd/msmd/splits/all_split.yaml \
     --config exp_configs/mutopia_full_aug_lc.yaml \
     -piece MozartWA__KV331__KV331_1_5_var4
"""
import os
import argparse
import yaml
try:
    import cPickle as pickle
except ImportError:
    import pickle
import theano
import numpy as np
import matplotlib.pyplot as plt
import cv2

import lasagne
from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD
from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model, compile_tag
from audio_sheet_retrieval.utils.batch_iterators import batch_compute1
import audio_sheet_retrieval.utils.video_rendering as vr
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data_video


def prepare_frames(specs, scores, atts):
    output_frames = []
    max_attention = np.max(atts)

    for cur_frame_idx in range(len(specs)):
        cur_spec = specs[cur_frame_idx]
        cur_score = scores[cur_frame_idx]
        cur_att = atts[cur_frame_idx] / max_attention

        cur_score_bgr = vr.prepare_img_for_render(cur_score, rsz_factor=1)
        cur_spec_bgr = vr.prepare_spec_for_render(cur_spec, rsz_factor=4)
        cur_att_bgr = vr.prepare_distribution_for_render(cur_att, width_rsz_factor=4)

        # initialize frame
        n_spacer = int(20)
        n_rows = cur_score_bgr.shape[0] + n_spacer + cur_att_bgr.shape[0] + n_spacer + cur_spec_bgr.shape[0]
        n_black_border = int(50)
        n_cols = n_black_border + cur_spec_bgr.shape[1] + n_black_border
        middle_col = int(n_cols / 2)
        cur_frame = np.ones((n_rows, n_cols, 3), dtype=np.uint8)

        # build frame
        cur_row_pointer = 0

        # sheet music
        start_idx = int(middle_col - cur_score_bgr.shape[1] / 2)
        end_idx = start_idx + cur_score_bgr.shape[1]
        cur_frame[cur_row_pointer:cur_row_pointer + cur_score_bgr.shape[0], start_idx:end_idx] = cur_score_bgr
        cur_row_pointer += cur_score_bgr.shape[0]
        cur_row_pointer += n_spacer

        # attention
        start_idx = n_black_border
        end_idx = start_idx + cur_att_bgr.shape[1]
        cur_frame[cur_row_pointer:cur_row_pointer + cur_att_bgr.shape[0], start_idx:end_idx] = cur_att_bgr
        cv2.putText(cur_frame, '{:.2f}'.format(max_attention), (15, cur_row_pointer),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(200, 200, 200), thickness=1)
        cv2.putText(cur_frame, '0', (30, cur_row_pointer + cur_att_bgr.shape[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(200, 200, 200), thickness=1)
        cur_row_pointer += cur_att_bgr.shape[0]
        cur_row_pointer += n_spacer

        # spectrogram
        start_idx = n_black_border
        end_idx = start_idx + cur_spec_bgr.shape[1]
        cur_frame[cur_row_pointer:cur_row_pointer + cur_spec_bgr.shape[0], start_idx:end_idx] = cur_spec_bgr

        output_frames.append(cur_frame)

    return output_frames


def main(args):
    # load config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)

    # load datapool with single piece
    print("\nLoading data...")
    audio_slices, sheet_slices, path_audio = prepare_piece_data_video(DATA_ROOT_MSMD, args.piece,
                                                                      fps=config['FPS'],
                                                                      sheet_context=config['SHEET_CONTEXT'],
                                                                      spec_context=config['SPEC_CONTEXT'])
    # select model
    model, _ = select_model(args.model)

    if not hasattr(model, 'prepare'):
        model.prepare = None

    print("Building network %s ..." % model.EXP_NAME)
    layers = model.build_model(input_shape_1=[1, config['STAFF_HEIGHT'], config['SHEET_CONTEXT']],
                               input_shape_2=[1, config['SPEC_BINS'], config['SPEC_CONTEXT']],
                               show_model=False)

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

    print("\nCompiling prediction functions...")
    l_view1, l_view2, l_v1latent, l_v2latent = layers

    # audio attention layer
    attention_layer = l_v2latent
    while attention_layer.name != "attention":
        if not hasattr(attention_layer, "input_layer"):
            attention_layer = attention_layer.input_layers[1]
        else:
            attention_layer = attention_layer.input_layer

    compute_attention = theano.function(inputs=[l_view2.input_var],
                                        outputs=lasagne.layers.get_output(attention_layer, deterministic=True))

    # iterate test data
    print("Computing attention...")

    # compute output on test set
    X1 = np.expand_dims(np.asarray(sheet_slices), axis=1).astype(np.float32)
    X2 = np.expand_dims(np.asarray(audio_slices), axis=1).astype(np.float32)

    # compute attention on audio input
    att = batch_compute1(X2, compute_attention, batch_size=100, verbose=False, prepare=None)

    plt.imshow(att, aspect='auto', origin='lower')
    plt.title('Attention Layer')
    plt.xlabel('Input Attention (bins)')
    plt.ylabel('Time (frames)')
    plt.colorbar()
    plt.savefig('{}_attention.png'.format(args.piece))

    # loop over frames
    print('Writing video...')
    output_frames = prepare_frames(audio_slices, sheet_slices, att)
    frame_rate = config['FPS']
    path_video = vr.write_video(output_frames, path_output='{}.mp4'.format(args.piece),
                                frame_rate=frame_rate, overwrite=True)
    vr.mux_video_audio(path_video, path_audio, path_output='{}_audio.mp4'.format(args.piece))


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Evaluate cross-modality retrieval model.')
    parser.add_argument('--model', help='select model to evaluate.')
    parser.add_argument('--data', help='select evaluation data.', type=str)
    parser.add_argument('--piece', help='piece name.', type=str)
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    parser.add_argument('--seed', help='random seed', type=int, default=23)
    args = parser.parse_args()

    main(args)
