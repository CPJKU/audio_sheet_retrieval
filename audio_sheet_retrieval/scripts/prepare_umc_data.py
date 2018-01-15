
import os
import cv2
import glob
import shutil
import numpy as np

# SRC_DIR = "/media/matthias/Data/Data/umc_export/"
# DST_DIR = "/home/matthias/Desktop/umc_chopin/"

# SRC_DIR = "/media/matthias/Data/umc_export/"
SRC_DIR = "/home/matthias/mounts/root@coltrane/srv/files/cp/temp/export_matthias/"
# DST_DIR = "/media/matthias/Data/umc_chopin/"
DST_DIR = "/media/matthias/Data/umc_beethoven/"


def resize_img(I):
    width = 835
    scale = float(width) / I.shape[1]
    height = int(scale * I.shape[0])
    I = cv2.resize(I, (width, height))
    return I


if __name__ == "__main__":
    """ main """

    # get list of pieces
    piece_dirs = np.sort(glob.glob(os.path.join(SRC_DIR, "*")))

    # iterate pieces
    for i_piece, piece_dir in enumerate(piece_dirs):
        piece_name = os.path.basename(piece_dir)
        print "processing piece (%d / %d) %s" % (i_piece + 1, len(piece_dirs), piece_name)

        # if "Chopin" not in piece_dir:
        #     continue

        # if "Mozart" not in piece_dir:
        #     continue

        if "Beethoven" not in piece_dir:
            continue

        # prepare folders
        dst_piece_dir = os.path.join(os.path.join(DST_DIR, piece_name))
        os.mkdir(dst_piece_dir)
        os.mkdir(os.path.join(dst_piece_dir, "sheet"))

        # resize images
        page_dir = os.path.join(piece_dir, "pages/*.*")
        page_files = np.sort(glob.glob(page_dir))
        for page_file in page_files:
            I = cv2.imread(page_file, 0)
            I = resize_img(I)

            file_name = os.path.basename(page_file).replace(".jpg", ".png")
            dst_img_file = os.path.join(dst_piece_dir, "sheet", file_name)
            cv2.imwrite(dst_img_file, I)

        # copy midi file
        src = os.path.join(piece_dir, "score_ppq.mid")
        dst_midi_file = os.path.join(dst_piece_dir, "score_ppq.mid")
        shutil.copy(src, dst_midi_file)

        # render audio
        from sheet_manager.render_audio import render_audio
        audio_path, perf_midi_path = render_audio(dst_midi_file, sound_font="grand-piano-YDP-20160804",
                                                  velocity=None, change_tempo=True, tempo_ratio=1.0,
                                                  target_dir=None, quiet=True, audio_fmt=".flac",
                                                  sound_font_root="~/.fluidsynth")
        trg_audio_path = os.path.join(dst_piece_dir, "score_ppq.flac")
        shutil.copy(audio_path, trg_audio_path)

        # copy performances if there
        perf_path = os.path.join(piece_dir, "audio")
        if os.path.exists(perf_path):
            audio_paths = glob.glob(os.path.join(perf_path, "*"))
            print "Found %d performances" % len(audio_paths)
            audio_paths = audio_paths[:3]
            for i_audio, audio_path in enumerate(audio_paths):
                ext = os.path.splitext(audio_path)[1]
                dst = os.path.join(dst_piece_dir, "%02d_performance%s" % (i_audio + 1, ext))
                shutil.copy(audio_path, dst)
