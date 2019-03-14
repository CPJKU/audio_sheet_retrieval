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

import lasagne
from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model, select_data, compile_tag
from audio_sheet_retrieval.utils.batch_iterators import batch_compute1
from audio_sheet_retrieval.utils.video_rendering import prepare_spec_for_render, prepare_distribution_for_render, write_video, mux_video_audio


def render_video(specs, atts):
    output_frames = []

    for cur_frame_idx in range(specs.shape[0]):
        cur_spec = specs[cur_frame_idx, 0]
        cur_att = atts[cur_frame_idx]

        cur_spec_bgr = prepare_spec_for_render(cur_spec, rsz_factor=4)
        cur_att_bgr = prepare_distribution_for_render(cur_att, width_rsz_factor=4)

        # initialize frame
        n_spacer = 20
        n_rows = cur_att_bgr.shape[0] + n_spacer + cur_spec_bgr.shape[0]
        n_cols = cur_spec_bgr.shape[1]
        cur_frame = np.ones((n_rows, n_cols, 3), dtype=np.uint8)

        # build frame
        cur_row_pointer = 0
        cur_frame[:cur_att_bgr.shape[0]] = cur_att_bgr
        cur_row_pointer += cur_att_bgr.shape[0]
        cur_row_pointer += n_spacer
        cur_frame[cur_row_pointer:cur_row_pointer + cur_spec_bgr.shape[0]] = cur_spec_bgr

        output_frames.append(cur_frame)

    return output_frames


def main(args):
    # load config
    with open(args.config, 'rb') as hdl:
        config = yaml.load(hdl)

    # load datapool with single piece
    print("\nLoading data...")
    eval_set = 'test'
    data = select_data(args.data, args.train_split, args.config, args.seed, test_only=True, piece_name=args.piece)

    # select model
    model, _ = select_model(args.model)

    if not hasattr(model, 'prepare'):
        model.prepare = None

    print("Building network %s ..." % model.EXP_NAME)
    layers = model.build_model(input_shape_1=[1, data[eval_set].staff_height, data[eval_set].sheet_context],
                               input_shape_2=[1, data[eval_set].spec_bins, data[eval_set].spec_context],
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
    n_test = data[eval_set].shape[0]
    indices = np.linspace(0, data[eval_set].shape[0] - 1, n_test).astype(np.int)
    _, X2 = data[eval_set][indices]

    # compute attention
    att = batch_compute1(X2, compute_attention, batch_size=100, verbose=False, prepare=None)

    import matplotlib.pyplot as plt
    plt.imshow(att, aspect='auto')
    plt.colorbar()
    plt.show()
    # loop over frames
    # print('Writing video...')
    # path_audio = data[eval_set].audio_pathes[0]
    # output_frames = render_video(X2, att)
    # path_video = write_video(output_frames, path_output='{}.mp4'.format(args.piece), overwrite=False)
    # mux_video_audio(path_video, path_audio, path_output='{}_audio.mp4'.format(args.piece))


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
    parser.add_argument('--seed', help='query direction.', type=int, default=23)
    args = parser.parse_args()

    main(args)
