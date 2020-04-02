
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

SHEET_WINDOW = 100
SPEC_WINDOW = 40


class ContinuousSpec2SheetHashingPool(object):
    """
    Data Pool for spectrogram to sheet snippet hashing
    """

    def __init__(self, sheets, coords, spectrograms, onsets, spec_context, sheet_context, staff_height=50,
                 shuffle=True):
        """
        Constructor
        """

        self.sheets = sheets
        self.coords = coords
        self.spectrograms = spectrograms
        self.onsets = onsets
        self.spec_context = spec_context
        self.sheet_context = sheet_context

        self.shape = None

        # get dimensions of inputs
        self.staff_height = staff_height
        self.sheet_dim = [self.staff_height, self.sheets[0].shape[1]]
        self.spec_dim = [self.spectrograms[0].shape[0], self.spec_context]

        # prepare train data
        self.train_entities = None
        self.prepare_train_entities()

        # shuffle data
        if shuffle:
            self.reset_batch_generator()

    def prepare_train_entities(self):
        """ collect train entities """

        self.train_entities = np.zeros((0, 2), dtype=np.int)

        # iterate sheets
        for i_sheet in xrange(len(self.sheets)):
            spec = self.spectrograms[i_sheet]
            sheet = self.sheets[i_sheet]

            o0 = self.spec_context // 2
            o1 = spec.shape[1] - self.spec_context // 2

            c0 = self.sheet_context // 2
            c1 = sheet.shape[1] - self.sheet_context // 2

            # iterate onsets in sheet
            for i_onset in xrange(0, len(self.onsets[i_sheet])):
                onset = self.onsets[i_sheet][i_onset]
                x_coord = self.coords[i_sheet][i_onset][1]
                if o0 < onset < o1 and c0 < x_coord < c1:
                    cur_entities = np.asarray([i_sheet, i_onset])
                    self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self, indices=None):
        """ reset batch generator """
        if indices is None:
            indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def __getitem__(self, key):
        """ make class accessible by index or slice """

        # get batch
        if key.__class__ != slice and key.__class__ != np.ndarray:
            key = slice(key, key + 1)
        batch_entities = self.train_entities[key]

        # collect train entities
        Sheet_batch = np.zeros((len(batch_entities), 1, self.sheet_dim[0], self.sheet_context), dtype=np.float32)
        Spec_batch = np.zeros((len(batch_entities), 1, self.spec_dim[0], self.spec_context), dtype=np.float32)
        for i_entity, (i_sheet, i_onset) in enumerate(batch_entities):
            # get sheet and annotations
            sheet = self.sheets[i_sheet]
            spec = self.spectrograms[i_sheet]
            sel_onset = int(self.onsets[i_sheet][i_onset])
            coords = self.coords[i_sheet]

            # get sliding window image snippet
            x = int(coords[i_onset, 1])
            x0 = x - self.sheet_context // 2
            x1 = x0 + self.sheet_context
            sliding_window = sheet[:, x0:x1]

            # get sliding window spectrogram excerpt
            t0 = sel_onset - self.spec_context // 2
            t1 = t0 + self.spec_context
            E = spec[:, t0:t1]

            # collect batch data
            Sheet_batch[i_entity, 0, :, :] = sliding_window
            Spec_batch[i_entity, 0, :, :] = E

        return Sheet_batch, Spec_batch


def align_baseline(dists):
    """ Compute alignment baseline by interpolation """
    i1_sheet = dists.shape[0]
    align_sheet_idxs = np.linspace(start=0, stop=i1_sheet - 1, num=dists.shape[1])
    return align_sheet_idxs


def align_pydtw(dists):
    """ Use python dtw package for alignment """
    from audio_sheet_retrieval.utils.dtw_by_dist import dtw_by_dist

    min_dist, C, C_acc, path = dtw_by_dist(dists)

    # fix path
    audio_path = []
    align_sheet_idxs = []
    for i in xrange(dists.shape[1]):
        sheet_idx = np.nonzero(path[0] == i)[0][0]
        audio_path.append(path[0][sheet_idx])
        align_sheet_idxs.append(path[1][sheet_idx])

    align_sheet_idxs = np.array(align_sheet_idxs)

    # plt.figure("DTW")
    # plt.imshow(dists, cmap=cmaps['magma'])
    # plt.plot(audio_path, align_sheet_idxs, 'c-', linewidth=3, alpha=0.5)
    # plt.show(block=True)

    return align_sheet_idxs


def compute_alignment(img_codes, spec_codes, sheet_idxs, spec_idxs, align_by):
    """ Evaluate Alignment """

    # compute distance matrix
    dists = cdist(img_codes, spec_codes, metric="cosine")

    # align audio part of distance matrix to sheet
    if align_by == 'baseline':
        aligned_sheet_idxs = align_baseline(dists)
    elif align_by == 'pydtw':
        aligned_sheet_idxs = align_pydtw(dists)
    else:
        pass

    # map matrix indices to coordinates
    aligned_sheet_idxs = np.round(aligned_sheet_idxs).astype(np.int)
    aligned_sheet_coords = sheet_idxs[aligned_sheet_idxs]

    # interpolate alignment
    filterd_idxs = np.diff(np.concatenate((spec_idxs[0:1] - 1, spec_idxs))) > 0
    f_inter = interp1d(spec_idxs[filterd_idxs], aligned_sheet_coords[filterd_idxs])
    i_inter = np.arange(spec_idxs[0], spec_idxs[-1] + 1, 1)
    a2s_alignment = f_inter(i_inter)

    # compute frame to coord mapping
    a2s_mapping = dict(zip(i_inter, a2s_alignment))

    dtw_res = {"dists": dists, "aligned_sheet_idxs": aligned_sheet_idxs,
               "aligned_sheet_coords": aligned_sheet_coords, "i_inter": i_inter,
               "a2s_alignment": a2s_alignment, "spec_idxs": spec_idxs}

    return a2s_mapping, dtw_res


def estimate_alignment_error(true_coords, true_onsets, a2s_mapping):
    """ Compute alignment error measures """

    pxl_errors = np.zeros(len(true_onsets))
    for j, o in enumerate(true_onsets):
        if o in a2s_mapping:
            look_up = int(o)
            pxl_errors[j] = true_coords[j] - a2s_mapping[look_up]

    return pxl_errors
