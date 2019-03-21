
from __future__ import print_function

import sys
import yaml
from tqdm import tqdm

from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool, AUGMENT, NO_AUGMENT


def load_split(split_file):
    """Helper function to load split YAML."""

    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl)

    return split


def load_piece_list(piece_names, raw_audio=False, aug_config=NO_AUGMENT, fps=20):
    """
    Collect piece data
    """
    all_images = []
    all_specs = []
    all_o2c_maps = []
    all_pathes_audio = []

    for ip in tqdm(range(len(piece_names)), ncols=70):
        piece_name = piece_names[ip]

        try:
            image, specs, o2c_maps, path_audio = prepare_piece_data(DATA_ROOT_MSMD, piece_name, raw_audio=raw_audio,
                                                                    aug_config=aug_config, require_audio=raw_audio,
                                                                    fps=fps)
        except KeyboardInterrupt:
            break
        except IndexError:
            print("{}: Performance not available.".format(piece_name))
            continue

        # keep stuff
        all_images.append(image)
        all_specs.append(specs)
        all_o2c_maps.append(o2c_maps)
        all_pathes_audio.append(path_audio)

    return all_images, all_specs, all_o2c_maps, all_pathes_audio


def load_audio_score_retrieval(split_file, config=None, test_only=False, piece_name=None):
    """Load alignment data in three AudioScoreRetrievalPools.

    Paramters
    ---------
    split_file : str
        Path to split file.
    config : dict
        Configurations for the datapools
    test_only : boolean
        Only create test-datapool
    piece_name : str
        Create test-datapool with a single piece only
        (used for video rendering).
    """

    if not config:
        spec_bins = None
        spec_context = None
        fps = 20
        sheet_context = None
        staff_height = None
        augment = AUGMENT
        no_augment = NO_AUGMENT
        test_augment = NO_AUGMENT.copy()
        raw_audio = False
    else:
        spec_context = config["SPEC_CONTEXT"]
        spec_bins = config["SPEC_BINS"]
        fps = config["FPS"]
        sheet_context = config["SHEET_CONTEXT"]
        staff_height = config["STAFF_HEIGHT"]
        augment = config["AUGMENT"]
        no_augment = NO_AUGMENT
        test_augment = NO_AUGMENT.copy()
        test_augment['synths'] = [config["TEST_SYNTH"]]
        test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]
        raw_audio = config["RAW_AUDIO"]

    # selected pieces
    split = load_split(split_file)

    # initialize data pools
    if not test_only:
        tr_images, tr_specs, tr_o2c_maps, tr_audio_pathes = load_piece_list(split['train'], aug_config=augment,
                                                                            raw_audio=raw_audio, fps=fps)
        tr_pool = AudioScoreRetrievalPool(tr_images, tr_specs, tr_o2c_maps, tr_audio_pathes,
                                          spec_context=spec_context, spec_bins=spec_bins,
                                          sheet_context=sheet_context, staff_height=staff_height,
                                          data_augmentation=augment, shuffle=True)
        print("Train: %d" % tr_pool.shape[0])

        va_images, va_specs, va_o2c_maps, va_audio_pathes = load_piece_list(split['valid'], aug_config=no_augment,
                                                                            raw_audio=raw_audio, fps=fps)
        va_pool = AudioScoreRetrievalPool(va_images, va_specs, va_o2c_maps, va_audio_pathes,
                                          spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                          data_augmentation=no_augment, shuffle=False)
        va_pool.reset_batch_generator()
        print("Valid: %d" % va_pool.shape[0])

    else:
        tr_pool = va_pool = None

    if piece_name is not None and test_only:
        split['test'] = [piece_name, ]

    te_images, te_specs, te_o2c_maps, te_audio_pathes = load_piece_list(split['test'], aug_config=test_augment,
                                                                        raw_audio=raw_audio, fps=fps)
    te_pool = AudioScoreRetrievalPool(te_images, te_specs, te_o2c_maps, te_audio_pathes,
                                      spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                      data_augmentation=no_augment, shuffle=False)
    print("Test: %d" % te_pool.shape[0])

    return dict(train=tr_pool, valid=va_pool, test=te_pool, train_tag="")


if __name__ == "__main__":
    """ main """
    # Profiling
    # import cProfile
    # profile_file = "profile.dmp"
    # print("trying to profile...")
    # print("output to: ", profile_file)
    # cProfile.run('main()', profile_file)

    import matplotlib.pyplot as plt
    RAW_AUDIO = True

    if RAW_AUDIO:
        config_file = '../exp_configs/mutopia_no_aug_raw.yaml'
    else:
        config_file = '../exp_configs/mutopia_no_aug.yaml'

    data = load_audio_score_retrieval(split_file="/media/rk1/home/stefanb/dev/msmd/msmd/splits/test_split.yaml",
                                      config_file=config_file,
                                      test_only=True)

    def train_batch_iterator(batch_size=1):
        """ Compile batch iterator """
        from audio_sheet_retrieval.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
        batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=None, k_samples=None)
        return batch_iterator

    bi = train_batch_iterator(batch_size=5)

    iterator = bi(data["test"])

    # show some train samples
    for epoch in range(10):
        for i, (sheet, audio_repr) in enumerate(iterator):

            plt.figure()
            plt.clf()

            plt.subplot(1, 2, 1)
            plt.imshow(sheet[0, 0], cmap="gray")
            plt.ylabel(sheet[0, 0].shape[0])
            plt.xlabel(sheet[0, 0].shape[1])
            # plt.colorbar()

            plt.subplot(1, 2, 2)

            if RAW_AUDIO:
                from msmd.midi_parser import extract_spectrogram
                spec = extract_spectrogram(audio_repr[0, 0, 0])
                # plt.plot(audio_repr[0, 0, 0])
                plt.imshow(spec, cmap="gray_r", origin="lower")
            else:
                plt.imshow(audio_repr[0, 0], cmap="gray_r", origin="lower")
            plt.ylabel(audio_repr[0, 0].shape[0])
            plt.xlabel(audio_repr[0, 0].shape[1])
            # plt.colorbar()
            plt.savefig('{}_{}.png'.format(epoch, i))
            # plt.show()
