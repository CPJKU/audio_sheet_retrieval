"""
    Create a video from a song contained in the MSMSD dataset.
"""
import numpy as np

from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data_video
import audio_sheet_retrieval.utils.video_rendering as vr


def prepare_frames(specs, scores):
    output_frames = []

    for cur_frame_idx in range(len(specs)):
        cur_spec = specs[cur_frame_idx]
        cur_score = scores[cur_frame_idx].astype(np.uint8)

        cur_score_bgr = vr.prepare_img_for_render(cur_score, rsz_factor=1)
        cur_spec_bgr = vr.prepare_spec_for_render(cur_spec, rsz_factor=4)

        # initialize frame
        n_spacer = int(20)
        n_rows = cur_score_bgr.shape[0] + n_spacer + cur_spec_bgr.shape[0]
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

        # spectrogram
        start_idx = n_black_border
        end_idx = start_idx + cur_spec_bgr.shape[1]
        cur_frame[cur_row_pointer:cur_row_pointer + cur_spec_bgr.shape[0], start_idx:end_idx] = cur_spec_bgr

        output_frames.append(cur_frame)

    return output_frames


if __name__ == '__main__':
    test_aug = dict()
    test_aug['system_translation'] = 0
    test_aug['sheet_scaling'] = [1.00, 1.00]
    test_aug['onset_translation'] = 0
    test_aug['spec_padding'] = 0
    test_aug['interpolate'] = -1
    test_aug['synths'] = ['grand-piano-YDP-20160804']
    test_aug['tempo_range'] = [2.0, 2.0]

    piece_name = 'BachJS__BWVAnh131__air'
    audio_slices, sheet_slices, path_audio = prepare_piece_data_video(DATA_ROOT_MSMD, piece_name, aug_config=test_aug,
                                                                      fps=20, sheet_context=200, spec_context=168)
    output_frames = prepare_frames(audio_slices, sheet_slices)
    frame_rate = 20
    path_video = vr.write_video(output_frames, path_output='{}.mp4'.format(piece_name),
                                frame_rate=frame_rate, overwrite=True)
    vr.mux_video_audio(path_video, path_audio, path_output='{}_audio.mp4'.format(piece_name))
