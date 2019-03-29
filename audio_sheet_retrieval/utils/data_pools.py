from __future__ import print_function
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from madmom.audio.signal import Signal
from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD

try:
    from msmd.midi_parser import notes_to_onsets
    from msmd.data_model.piece import Piece
    from msmd.alignments import align_score_to_performance
except ImportError:
    raise ImportError('Could not import msmd dataset package. Please install!')


NO_AUGMENT = dict()
NO_AUGMENT['system_translation'] = 0
NO_AUGMENT['sheet_scaling'] = [1.00, 1.00]
NO_AUGMENT['onset_translation'] = 0
NO_AUGMENT['spec_padding'] = 0
NO_AUGMENT['interpolate'] = -1
NO_AUGMENT['synths'] = ['ElectricPiano']
NO_AUGMENT['tempo_range'] = [1.00, 1.00]

# this will be overwritten with a config file
# (see audio_sheet_retrieval/exp_configs)
AUGMENT = dict()
for key in NO_AUGMENT.keys():
    AUGMENT[key] = NO_AUGMENT[key]


class AudioScoreRetrievalPool(object):

    def __init__(self, images, specs, o2c_maps, audio_pathes,
                 spec_context=None, spec_bins=None, sheet_context=None, staff_height=None,
                 data_augmentation=None, shuffle=True, mode='training', return_piece_names=False,
                 return_n_onsets=False):

        if spec_context is None:
            spec_context = 42

        if spec_bins is None:
            spec_bins = 92

        if sheet_context is None:
            sheet_context = 200

        if staff_height is None:
            staff_height = 160

        if data_augmentation is None:
            data_augmentation = NO_AUGMENT

        self.images = images
        self.specs = specs
        self.o2c_maps = o2c_maps
        self.audio_pathes = audio_pathes
        self.mode = mode
        self.return_piece_names = return_piece_names
        self.return_n_onsets = return_n_onsets

        self.spec_context = spec_context
        self.spec_bins = spec_bins
        self.sheet_context = sheet_context
        self.staff_height = staff_height

        self.data_augmentation = data_augmentation
        self.shuffle = shuffle

        self.shape = None
        self.sheet_dim = [self.staff_height, self.sheet_context]
        self.audio_dim = [self.specs[0][0].shape[0], self.spec_context]

        if self.data_augmentation['interpolate'] > 0:
            self.interpolate()

        self.prepare_train_entities()

        if self.shuffle:
            self.reset_batch_generator()

    def interpolate(self):
        """
        Interpolate onset to note correspondences on frame level
        """
        for i_sheet in range(len(self.images)):
            for i_spec in range(len(self.specs[i_sheet])):

                onsets = self.o2c_maps[i_sheet][i_spec][:, 0]
                coords = self.o2c_maps[i_sheet][i_spec][:, 1]

                # interpolate some extra onsets
                step_size = self.data_augmentation['interpolate']
                f_inter = interp1d(onsets, coords)
                onsets = np.arange(onsets[0], onsets[-1], step_size)
                coords = f_inter(onsets)

                # update mapping
                onsets = onsets.reshape((-1, 1))
                coords = coords.reshape((-1, 1))
                new_mapping = np.hstack((onsets, coords))
                self.o2c_maps[i_sheet][i_spec] = new_mapping.astype(np.int)

    def prepare_train_entities(self):
        """
        Collect train entities
        """

        self.train_entities = np.zeros((0, 3), dtype=np.int)

        # iterate sheets
        for i_sheet, sheet in enumerate(self.images):

            # iterate spectrograms
            for i_spec, spec in enumerate(self.specs[i_sheet]):

                # iterate onsets in sheet
                for i_onset in range(len(self.o2c_maps[i_sheet][i_spec])):
                    onset = self.o2c_maps[i_sheet][i_spec][i_onset, 0]
                    o_start = onset - self.spec_context // 2
                    o_stop = o_start + self.spec_context

                    coord = self.o2c_maps[i_sheet][i_spec][i_onset, 1]
                    c_start = coord - self.sheet_context // 2
                    c_stop = c_start + self.sheet_context

                    if self.mode == 'training':
                        # only select samples which lie in the valid borders
                        if o_start >= 0 and o_stop < spec.shape[1]\
                                and c_start >= 0 and c_stop < sheet.shape[1]:
                            cur_entities = np.asarray([i_sheet, i_spec, i_onset])
                            self.train_entities = np.vstack((self.train_entities, cur_entities))

                    if self.mode == 'all':
                        cur_entities = np.asarray([i_sheet, i_spec, i_onset])
                        self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self):
        """
        Reset data pool
        """
        indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def prepare_train_image(self, i_sheet, i_spec, i_onset):
        """ prepare train item """

        # get sheet and annotations
        sheet = self.images[i_sheet]

        # get target note coodinate
        target_coord = self.o2c_maps[i_sheet][i_spec][i_onset][1]

        # get sub-image (with coordinate fixing)
        # this is done since we do not want to do the augmentation
        # on the whole sheet image
        c0 = max(0, target_coord - 2 * self.sheet_context)
        c1 = min(c0 + 4 * self.sheet_context, sheet.shape[1])
        c0 = max(0, c1 - 4 * self.sheet_context)
        sheet = sheet[:, c0:c1]

        if self.data_augmentation['sheet_scaling']:
            import cv2
            sc = self.data_augmentation['sheet_scaling']
            scale = (sc[1] - sc[0]) * np.random.random_sample() + sc[0]
            new_size = (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale))
            sheet = cv2.resize(sheet, new_size, interpolation=cv2.INTER_NEAREST)

        # target coordinate
        x = sheet.shape[1] // 2

        # compute sliding window coordinates
        x0 = int(np.max([x - self.sheet_context // 2, 0]))
        x1 = int(np.min([x0 + self.sheet_context, sheet.shape[1] - 1]))
        x0 = int(x1 - self.sheet_context)

        # get vertical crop
        r0 = sheet.shape[0] // 2 - self.staff_height // 2
        if self.data_augmentation['system_translation']:
            t = self.data_augmentation['system_translation']
            r0 += np.random.randint(low=-t, high=t + 1)
        r1 = r0 + self.staff_height

        # get sheet snippet
        sheet_snippet = sheet[r0:r1, x0:x1]

        return sheet_snippet

    def prepare_train_audio(self, i_sheet, i_spec, i_onset):
        """
        Prepare audio excerpt
        """

        # get spectrogram and onset
        spec = self.specs[i_sheet][i_spec]
        sel_onset = int(self.o2c_maps[i_sheet][i_spec][i_onset][0])

        # data augmentation note position
        if self.data_augmentation['onset_translation']:
            t = self.data_augmentation['onset_translation']
            sel_onset += np.random.randint(low=-t, high=t + 1)

        # compute sliding window coordinates
        start = np.max([sel_onset - self.spec_context // 2, 0])
        stop = start + self.spec_context

        stop = np.min([stop, spec.shape[1] - 1])
        start = stop - self.spec_context

        excerpt = spec[:, start:stop]

        if self.data_augmentation['spec_padding']:
            spec_padding = self.data_augmentation['spec_padding']
            excerpt = np.pad(excerpt, ((spec_padding, spec_padding), (0, 0)), mode='edge')
            s = np.random.randint(0, spec_padding)
            e = s + spec.shape[0]
            excerpt = excerpt[s:e, :]

        return excerpt

    def __getitem__(self, key):
        """
        Make class accessible by index or slice
        """

        # get batch
        if key.__class__ == int:
            key = slice(key, key + 1)

        batch_entities = self.train_entities[key]

        # collect train entities
        sheet_batch = np.zeros((len(batch_entities), 1, self.sheet_dim[0], self.sheet_context), dtype=np.float32)
        spec_batch = np.zeros((len(batch_entities), 1, self.audio_dim[0], self.spec_context), dtype=np.float32)
        piece_names_batch = []
        n_onsets = []

        for i_entity, (i_sheet, i_spec, i_onset) in enumerate(batch_entities):

            # get sliding window train item
            snippet = self.prepare_train_image(i_sheet, i_spec, i_onset)

            # get spectrogram excerpt (target note in center)
            excerpt = self.prepare_train_audio(i_sheet, i_spec, i_onset)

            if self.return_n_onsets:
                onsets_diff = np.abs(self.o2c_maps[i_sheet][i_spec][:, 0] - self.o2c_maps[i_sheet][i_spec][i_onset][0])
                # print(onsets_diff[i_onset-5:i_onset+5])

                n_onset = np.where(onsets_diff <= self.spec_context / 2)[0].shape[0]
                n_onsets.append(n_onset)

            # get corresponding piece name
            if self.return_piece_names:
                piece_name = os.path.basename(self.audio_pathes[i_sheet])
                piece_names_batch.append(piece_name)

            # collect batch data
            sheet_batch[i_entity, 0, :, :] = snippet
            spec_batch[i_entity, 0, :, :] = excerpt

        output = [sheet_batch, spec_batch]

        if self.return_piece_names:
            output.append(piece_names_batch)
        if self.return_n_onsets:
            output.append(n_onsets)

        return output


def onset_to_coordinates(alignment, mdict, note_events, fps):
    """
    Compute onset to coordinate mapping
    """
    onset_to_coord = np.zeros((0, 2), dtype=np.int)

    for m_objid, e_idx in alignment:

        # get note mungo and midi note event
        m, e = mdict[m_objid], note_events[e_idx]

        # compute onset frame
        onset_frame = notes_to_onsets([e], dt=1.0 / fps)

        # get note coodinates
        cy, cx = m.middle

        # keep mapping
        entry = np.asarray([onset_frame, cx], dtype=np.int)[np.newaxis]
        if onset_frame not in onset_to_coord[:, 0]:
            onset_to_coord = np.concatenate((onset_to_coord, entry), axis=0)

    return onset_to_coord


def systems_to_rois(sys_mungos, window_top=10, window_bottom=10):
    """
    Convert systems to rois
    """

    page_rois = np.zeros((0, 4, 2))
    for sys_mungo in sys_mungos:
        t, l, b, r = sys_mungo.bounding_box

        cr = (t + b) // 2

        r_min = cr - window_top
        r_max = r_min + window_top + window_bottom
        c_min = l
        c_max = r

        topLeft = [r_min, c_min]
        topRight = [r_min, c_max]
        bottomLeft = [r_max, c_min]
        bottomRight = [r_max, c_max]
        system = np.asarray([topLeft, topRight, bottomRight, bottomLeft])
        system = system.reshape((1, 4, 2))
        page_rois = np.vstack((page_rois, system))

    return page_rois.astype(np.int)


def stack_images(images, mungos_per_page, mdict):
    """
    Re-stitch image
    """
    stacked_image = images[0]
    stacked_page_mungos = [m for m in mungos_per_page[0]]

    row_offset = stacked_image.shape[0]

    for i in range(1, len(images)):

        # append image
        stacked_image = np.concatenate((stacked_image, images[i]))

        # update coordinates
        page_mungos = mungos_per_page[i]
        for m in page_mungos:
            m.x += row_offset
            stacked_page_mungos.append(m)
            mdict[m.objid] = m

        # update row offset
        row_offset = stacked_image.shape[0]

    return stacked_image, stacked_page_mungos, mdict


def unwrap_sheet_image(image, system_mungos, mdict, window_top=100, window_bottom=100):
    """
    Unwrap all systems of sheet image to a single "long row"
    """

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

    return un_wrapped_image, un_wrapped_coords


def prepare_piece_data(collection_dir, piece_name, aug_config=NO_AUGMENT,
                       raw_audio=False, require_audio=True, load_midi_matrix=False,
                       fps=20):
    """Load audio, sheet, and alignment data.

    Parameters
    ----------
    collection_dir : str
    piece_name : str
    aug_config : dict
        Augmentation settings
    raw_audio : bool
        Return raw audio instead of spectrograms.
    require_audio : bool
        If true, msmd checks for the respective audio files.
    load_midi_matrix : bool
        Load (and return) piano-roll representation of the MIDI.
    fps : int
        Frame rate in frames per second.

    Returns
    -------
    un_wrapped_image :
    audio_repr :
    onset_to_coord_maps :
    midi_matrices :
        Only returned if load_midi_matrix=True.
    path_audio : str
        Path to audio recording.
    """

    if raw_audio:
        require_audio = True

    # piece loading
    piece = Piece(root=collection_dir, name=piece_name)
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
    un_wrapped_image, un_wrapped_coords = unwrap_sheet_image(image, system_mungos, mdict)

    # load performances
    audio_repr = []
    midi_matrices = []
    onset_to_coord_maps = []

    for performance_key in piece.available_performances:

        # check if performance matches augmentation pattern
        tempo, synth = performance_key.split("tempo-")[1].split("_", 1)
        tempo = float(tempo) / 1000

        if synth not in aug_config["synths"]\
                or tempo < aug_config["tempo_range"][0]\
                or tempo > aug_config["tempo_range"][1]:
            continue

        # load current performance
        performance = piece.load_performance(performance_key, require_audio=require_audio)
        path_audio = performance.audio

        # load existing alignment from mung file
        alignment = piece.load_alignment(performance_key)

        try:
            assert len(alignment) > 0
        except AssertionError:
            print('{}: No alignment in Mung file. Please check your MSMD data.'.format(performance_key))

        # note events
        note_events = performance.load_note_events()

        if raw_audio:
            # load raw audio
            SAMPLE_RATE = 22050
            sig = Signal(performance.audio, num_channels=1, sample_rate=SAMPLE_RATE, dtype=np.float32)
            audio_repr.append(np.atleast_2d(sig))
        else:
            # load spectrogram
            spec = performance.load_spectrogram()
            audio_repr.append(spec)

        # compute onset to coordinate mapping
        onset_to_coord = onset_to_coordinates(alignment, un_wrapped_coords, note_events, fps=fps)
        onset_to_coord_maps.append(onset_to_coord)

        if load_midi_matrix:
            midi = performance.load_midi_matrix()
            midi_matrices.append(midi)

    if load_midi_matrix:
        return un_wrapped_image, audio_repr, onset_to_coord_maps, midi_matrices, path_audio
    else:
        return un_wrapped_image, audio_repr, onset_to_coord_maps, path_audio


def prepare_piece_data_video(collection_dir, piece_name, fps=20, sheet_context=200, spec_context=168):
    """Load audio and sheet pairs for a piece.

    For a single piece, get the data from MSMD and create
    lists of corresponding audio-sheet pairs as a list.

    Parameters
    ----------
    collection_dir : str
        Path to the MSMD dataset.
    piece_name : str
        Name of the piece
    fps : int
        Frame rate in frames per second.
    sheet_contex : int
        Number of context frames for the sheet.
    spec_context : int
        Number of context frames for the audio.

    Returns
    -------
    audio_slices : list
    sheet_slices : list
    path_audio : str
        Path to audio recording.
    """

    # Params from config
    SHEET_CONTEXT = sheet_context
    N_ZEROS_SHEET = SHEET_CONTEXT // 2
    SPEC_CONTEXT = spec_context
    N_ZEROS_SPEC = SPEC_CONTEXT // 2

    # get data from MSMD
    sheet_repr, audio_repr, onset_to_coord_maps, path_audio = \
        prepare_piece_data(collection_dir, piece_name, raw_audio=False,
                           require_audio=True, load_midi_matrix=False, fps=fps)
    audio_repr = audio_repr[0]
    onset_to_coord_maps = onset_to_coord_maps[0]
    n_audio_frames = audio_repr.shape[1]

    # pad with zeros
    sheet_repr = np.c_[np.zeros((sheet_repr.shape[0], N_ZEROS_SHEET)),
                       sheet_repr,
                       np.zeros((sheet_repr.shape[0], N_ZEROS_SHEET))]
    audio_repr = np.c_[np.zeros((audio_repr.shape[0], N_ZEROS_SPEC)),
                       audio_repr,
                       np.zeros((audio_repr.shape[0], N_ZEROS_SPEC))]

    # offset annotations and create interpolation function
    onset_to_coord_maps[:, 0] += N_ZEROS_SPEC
    onset_to_coord_maps[:, 1] += N_ZEROS_SHEET
    f_inter = interp1d(onset_to_coord_maps[:, 0], onset_to_coord_maps[:, 1])
    onsets = np.arange(onset_to_coord_maps[0, 0], onset_to_coord_maps[-1, 0], 1)
    coords = f_inter(onsets).astype(int)

    # fill with start and end linearly
    onsets = np.r_[np.arange(onsets[0]), onsets, np.arange(onsets[-1] + 1, audio_repr.shape[1])]
    coords = np.r_[np.linspace(N_ZEROS_SHEET, onset_to_coord_maps[0, 1] - 1, onset_to_coord_maps[0, 0] - 1),
                   coords]
    coords = np.r_[coords,
                   np.linspace(onset_to_coord_maps[-1, 1] + 1,
                               sheet_repr.shape[1],
                               onsets.shape[0] - coords.shape[0])].astype(int)

    # slice audio into input segments
    audio_slices = []
    sheet_slices = []

    for cur_start_idx in range(N_ZEROS_SPEC, n_audio_frames + N_ZEROS_SPEC):
        cur_slice = audio_repr[:, cur_start_idx - N_ZEROS_SPEC:cur_start_idx + N_ZEROS_SPEC]
        cur_sheet = sheet_repr[:, coords[cur_start_idx] - N_ZEROS_SHEET:coords[cur_start_idx] + N_ZEROS_SHEET]

        audio_slices.append(cur_slice)
        sheet_slices.append(cur_sheet)

    return audio_slices, sheet_slices, path_audio


def load_audio_score_retrieval_test(collection_dir):
    """
    Load alignment data
    """

    piece_names = ['BachCPE__cpe-bach-rondo__cpe-bach-rondo', 'BachJS__BWV259__bwv-259']

    all_piece_images = []
    all_piece_specs = []
    all_piece_o2c_maps = []
    all_piece_audio_pathes = []

    for piece_name in piece_names:

        piece_image, piece_specs, piece_o2c_maps, piece_audio_path = prepare_piece_data(collection_dir, piece_name)

        # keep stuff
        all_piece_images.append(piece_image)
        all_piece_specs.append(piece_specs)
        all_piece_o2c_maps.append(piece_o2c_maps)
        all_piece_audio_pathes.append(piece_audio_path)

    return AudioScoreRetrievalPool(all_piece_images, all_piece_specs, all_piece_o2c_maps, all_piece_audio_pathes,
                                   data_augmentation=AUGMENT, return_piece_names=True, return_n_onsets=True)


if __name__ == "__main__":
    """ main """

    pool = load_audio_score_retrieval_test(DATA_ROOT_MSMD)

    for i in range(10):
        sheet, spec, piece_name, n_onsets = pool[i:i+1]

        plt.figure()
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(sheet[0, 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(spec[0, 0], cmap='viridis', origin='lower')
        plt.suptitle('{}\n#Onsets: {}'.format(piece_name[0], n_onsets[0]))
        plt.savefig('{}.png'.format(i))
