"""
    Helper functions for rendering videos using OpenCV.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def prepare_spec_for_render(spec, rsz_factor=6):
    spec_prep = np.flipud(spec)

    # resize
    spec_prep = cv2.resize(spec_prep, (spec.shape[1] * rsz_factor, spec.shape[0] * rsz_factor))

    # get rgb version
    spec_prep = plt.cm.viridis(spec_prep)[:, :, 0:3]

    # convert to uint8
    spec_prep = (spec_prep * 255).astype(np.uint8)

    # convert to bgr (openCV standard)
    spec_prep = cv2.cvtColor(spec_prep, cv2.COLOR_RGB2BGR)

    return spec_prep


def prepare_distribution_for_render(dist, height=100, width_rsz_factor=1):
    prob_box = np.ones((height, int(width_rsz_factor * len(dist)), 3), dtype=np.uint8)
    max_len = int(height)

    # show top labels
    for cur_bar_idx in range(len(dist)):
        # plot probability bar
        cur_line_length = int(np.round(dist[cur_bar_idx] * max_len))

        col_coord = cur_bar_idx
        cv2.line(prob_box,
                 (width_rsz_factor * col_coord, prob_box.shape[0]),
                 (width_rsz_factor * col_coord, prob_box.shape[0] - cur_line_length),
                 (255, 255, 0), thickness=5)

    return prob_box


def write_video(images, path_output='output.mp4', frame_rate=20, overwrite=False):
    """Takes a list of images and interprets them as frames for a video.

    Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    height, width, _ = images[0].shape

    if overwrite:
        if os.path.exists(path_output):
            os.remove(path_output)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_output, fourcc, frame_rate, (width, height))

    for cur_image in images:
        frame = cv2.resize(cur_image, (width, height))
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()

    return path_output


def mux_video_audio(path_video, path_audio, path_output='output_audio.mp4'):
    """Use FFMPEG to mux video with audio recording."""

    ffmpeg_call = 'ffmpeg -y -i "{}" -i "{}" -shortest {}'.format(path_video, path_audio, path_output)
    os.system(ffmpeg_call)
