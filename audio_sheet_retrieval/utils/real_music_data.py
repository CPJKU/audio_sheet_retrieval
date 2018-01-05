
import os
import glob
import numpy as np
from scipy.interpolate import interp1d

from multi_modality_hashing.config.settings import DATA_ROOT_REAL_MUSIC_SF as ROOT_DIR

from omr.utils.data import MOZART_PIECES as PIECES
tr_pieces = PIECES
va_piece = "Mozart_Piano_Sonata_K545_1"

if va_piece in tr_pieces:
    tr_pieces.remove("Mozart_Piano_Sonata_K545_1")

va_pieces = [va_piece]

from omr.utils.data import BACH_PIECES, HAYDN_PIECES, BEETHOVEN_PIECES, CHOPIN_PIECES, STRAUSS_PIECES, SCHUBERT_PIECES
te_pieces = BACH_PIECES + HAYDN_PIECES + BEETHOVEN_PIECES + CHOPIN_PIECES + STRAUSS_PIECES + SCHUBERT_PIECES

SHEET_CONTEXT = 200
SYSTEM_HEIGHT = 180
SPEC_CONTEXT = 42
SPEC_BINS = 92

AUGMENT = dict()
AUGMENT['system_translation'] = 5  # 0 / 5
AUGMENT['sheet_scaling'] = [0.95, 1.05]  # [1.0, 1.0] / [0.95, 1.05]
AUGMENT['note_coord_translation'] = 5  # 0 / 5
AUGMENT['onset_translation'] = 1
AUGMENT['spec_padding'] = 3
AUGMENT['interpolate'] = -1

NO_AUGMENT = dict()
for key in AUGMENT.keys():
    NO_AUGMENT[key] = False

# TR_TEMPI = [900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100]
TR_TEMPI = [1000]
TE_TEMPI = [1000]

# TR_SOUNDFONTS = ['Unison', 'Acoustic_Piano', 'FluidR3_GM']
TR_SOUNDFONTS = ['Acoustic_Piano']
VA_SOUNDFONTS = ['Acoustic_Piano']
TE_SOUNDFONTS = ['Steinway']


class AugmentedContinousSpec2SheetHashingPool(object):
    """
    Data Pool for spectrogram to sheet snippet hashing
    """

    def __init__(self, sheets, coords, system_pos, spectrograms, onsets,
                 spec_context, sheet_context, staff_height=40, shuffle=True,
                 data_augmentation=None, roi_note_coords=None):
        """
        Constructor
        """

        self.sheets = sheets
        self.coords = coords
        self.system_pos = system_pos
        self.spectrograms = spectrograms
        self.onsets = onsets
        self.spec_context = spec_context
        self.sheet_context = sheet_context
        self.data_augmentation = data_augmentation

        self.shape = None
        
        # list of notes in image and onsets in spectrogam excerpts
        self.pair_counts = []        
        
        # get dimensions of inputs
        self.staff_height = staff_height
        self.sheet_dim = [self.staff_height, self.sheets[0].shape[1]]
        self.spec_dim = [self.spectrograms[0].shape[0], self.spec_context]
        
        # prepare train data
        if roi_note_coords is None:
            self.roi_note_coords = None
            self.prepare_sheet_image_data()
        else:
            self.roi_note_coords = roi_note_coords

        # prepare train instances
        self.prepare_train_entities()

        # shuffle data
        if shuffle:
            self.reset_batch_generator()
    
    def prepare_sheet_image_data(self):
        """ prepare sheet image data for faster iterations """
        
        self.roi_note_coords = []
        window = int(0.9 * SYSTEM_HEIGHT / 2)
        for i_sheet in xrange(len(self.sheets)):
            
            # get data
            sheet = self.sheets[i_sheet]
            systems = self.system_pos[i_sheet]
            note_coords = self.coords[i_sheet]
            onsets = self.onsets[i_sheet]
            
            # convert systems to regions of interrest
            rois = systems_to_rois(systems, window_top=window, window_bottom=window)
            
            # group note coordinates by rois
            roi_note_coords = group_by_roi(note_coords, rois)
            
            # unwrap data
            system_heigth = SYSTEM_HEIGHT + 40
            un_wrapped_sheet, un_wrapped_coords, un_wrapped_systems, un_wrapped_note_roi_coords = self.unwrap_sheet_images(sheet, systems, roi_note_coords, system_heigth, 1.0)

            # interpolate some extra onsets
            if self.data_augmentation['interpolate'] > 0:
                step_size = self.data_augmentation['interpolate']
                f_inter = interp1d(onsets, un_wrapped_coords[:, 1])
                onsets = np.arange(onsets[0], onsets[-1] + 1, step_size)
                coords_1 = f_inter(onsets).reshape((-1, 1))
                coords_0 = np.ones((len(onsets), 1), dtype=np.float32) * un_wrapped_sheet.shape[0] / 2
                un_wrapped_coords = np.hstack((coords_0, coords_1))
                un_wrapped_coords = np.around(un_wrapped_coords, 1)

                # re-group coordinates
                rois = un_wrapped_systems
                un_wrapped_note_roi_coords = group_by_roi(un_wrapped_coords, rois)

                # plt.figure()
                # plt.imshow(un_wrapped_sheet, cmap=plt.cm.gray)
                # plt.plot(un_wrapped_systems[:, :, 1], un_wrapped_systems[:, :, 0], 'mo')
                # plt.plot(un_wrapped_coords[:, 1], un_wrapped_coords[:, 0], 'co')
                # plt.show(block=True)
            
            # update to unwraped data
            self.sheets[i_sheet] = un_wrapped_sheet
            self.system_pos[i_sheet] = un_wrapped_systems
            self.onsets[i_sheet] = onsets
            self.coords[i_sheet] = un_wrapped_coords
            self.roi_note_coords.append(un_wrapped_note_roi_coords)
    
    def prepare_train_entities(self):
        """ collect train entities """

        self.train_entities = np.zeros((0, 2), dtype=np.int)

        # iterate sheets
        for i_sheet in xrange(len(self.sheets)):
            spec = self.spectrograms[i_sheet]

            # iterate onsets in sheet
            for i_onset in xrange(0, len(self.onsets[i_sheet])):
                onset = self.onsets[i_sheet][i_onset]
                start = onset - self.spec_context // 2
                stop = start + self.spec_context
                if start >= 0 and stop < spec.shape[1]:
                    cur_entities = np.asarray([i_sheet, i_onset])
                    self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self, indices=None):
        """ reset batch generator """
        if indices is None:
            indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def prepare_train_item(self, sheet, note_roi_coords, i_onset, systems):
        """ prepare train item """
        systems = systems.copy()
        note_roi_coords = [c.copy() for c in note_roi_coords]
        
        # get sub image
        note_roi_counts = np.cumsum([c.shape[0] for c in note_roi_coords])
        target_system_idx = np.nonzero((note_roi_counts >= i_onset))[0][0]
        system_idx0 = np.max([0, target_system_idx-1])
        system_idx1 = np.min([target_system_idx + 1, systems.shape[0] - 1])
        
        # get subimage
        c0 = int(systems[system_idx0, 0, 1])
        c1 = int(systems[system_idx1, 1, 1])
        sub_image = sheet[:, c0:c1]      
        
        # get sub image system coordinates
        sub_systems = systems[system_idx0:system_idx1+1].copy()
        sub_systems[:, :, 1] -= c0
        
        # get subimage note coordinates
        sub_note_roi_coords = []
        for i_sys in xrange(system_idx0, system_idx1+1):
            c = note_roi_coords[i_sys].copy()
            c[:, 1] -= c0
            sub_note_roi_coords.append(c)
        
#        plt.figure()
#        plt.imshow(sheet, cmap=plt.cm.gray)
#        plt.plot(systems[target_system_idx, :, 1], systems[target_system_idx, :, 0], 'mo')
#        
#        plt.figure()
#        plt.imshow(sub_image, cmap=plt.cm.gray)
#        plt.plot(sub_systems[:, :, 1], sub_systems[:, :, 0], 'mo')
#        for n in sub_note_roi_coords:
#            plt.plot(n[:, 1], n[:, 0], 'co')
#        
#        plt.show(block=True)
        
        # adopt data to image patch
        if system_idx0 > 0:
            i_onset -= note_roi_counts[system_idx0 - 1]
        sheet = sub_image
        systems = sub_systems
        note_roi_coords = sub_note_roi_coords

        if self.data_augmentation['system_translation']:
            t = self.data_augmentation['system_translation']
            for i in xrange(systems.shape[0]):
                systems[i, :, 0] += np.random.randint(low=-t, high=t+1)

        if self.data_augmentation['sheet_scaling']:
            import cv2
            sc = self.data_augmentation['sheet_scaling']
            scale = (sc[1] - sc[0]) * np.random.random_sample() + sc[0]
            for i in xrange(len(note_roi_coords)):
                note_roi_coords[i] *= scale
            systems *= scale
            new_size = (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale))
            sheet = cv2.resize(sheet, new_size, interpolation=cv2.INTER_NEAREST)
        
        # re-stitch systems
        stitched_image, stitched_coords = self.stitch_image(sheet, systems, note_roi_coords, SYSTEM_HEIGHT)
        
        # target coordinate
        x = int(stitched_coords[i_onset, 1])
        
        if self.data_augmentation['note_coord_translation']:
            t = self.data_augmentation['note_coord_translation']
            shift = np.random.randint(low=-t, high=t+1)
            x += shift
        
        # compute sliding window coordinates
        x0 = np.max([x - self.sheet_context // 2, 0])
        x1 = x0 + self.sheet_context
        
        x1 = np.min([x1, stitched_image.shape[1] - 1])
        x0 = x1 - self.sheet_context
        
        # get sliding window
        sliding_window = stitched_image[:, x0:x1]
        
        # compute notes contained in sliding window
        contained_notes = np.sum((stitched_coords[:, 1] > x0)  & (stitched_coords[:, 1] < x1))
        
        return sliding_window, contained_notes
    
    def unwrap_sheet_images(self, image, systems, notes, system_heigth, scale):
        """
        Unwrap all sytems of sheet image to a single "long row"
        """
        
        x_offset = 0
        n_notes = 0
        for n in notes:
            n_notes += n.shape[0]
        
        un_wrapped_roi_coords = []
        un_wrapped_coords = np.zeros((n_notes, 2))
        un_wrapped_systems = np.zeros_like(systems)
        
        width = image.shape[1] * systems.shape[0]
        window = int(system_heigth / 2 * scale)
        un_wrapped_image = np.zeros((2 * window, width), dtype=np.uint8)
        
        rois = systems_to_rois(systems, window_top=window, window_bottom=window)

        # make single staff image
        img_start = 0
        notes_start = 0
        for j, r in enumerate(rois):
            # image
            system_image = image[r[0, 0]:r[3, 0], r[0, 1]:r[1, 1]]
            img_end = img_start + system_image.shape[1]
            un_wrapped_image[:, img_start:img_end] = system_image

            # notes
            unw_notes = notes[j].copy()
            unw_system = systems[j].copy()
            
            unw_notes[:, 1] += -r[0, 1] + x_offset
            unw_system[:, 1] += -r[0, 1] + x_offset
            
            x_offset += (r[1, 1] - r[0, 1])
            
            unw_notes[:, 0] -= r[0, 0]
            unw_system[:, 0] -= r[0, 0]
            
            notes_end = notes_start + unw_notes.shape[0]
            un_wrapped_coords[notes_start:notes_end] = unw_notes
            un_wrapped_roi_coords.append(unw_notes)
            un_wrapped_systems[j] = unw_system
            
            img_start = img_end
            notes_start = notes_end
        
        # get relevant part of unwrapped image
        un_wrapped_image = un_wrapped_image[:, :img_end]
        
        # clip or pad image in case of scaling
        if scale > 1.0:
            r0 = (un_wrapped_image.shape[0] - system_heigth) // 2
            r1 = r0 + system_heigth
            un_wrapped_image = un_wrapped_image[r0:r1, :]
            un_wrapped_coords[:, 0] -= r0
            un_wrapped_systems[:, 0] -= r0
            for i in xrange(len(un_wrapped_roi_coords)):
                un_wrapped_roi_coords[i][:, 0] -= r0
        
        elif scale < 1.0:
            missing_top = (system_heigth - un_wrapped_image.shape[0]) // 2
            missing_bottom = system_heigth - un_wrapped_image.shape[0] - missing_top
            un_wrapped_image = np.pad(un_wrapped_image, ((missing_top, missing_bottom), (0, 0)), mode='edge')
            un_wrapped_coords[:, 0] += missing_top
            un_wrapped_systems[:, 0] += missing_top
            for i in xrange(len(un_wrapped_roi_coords)):
                un_wrapped_roi_coords[i][:, 0] += missing_top
        
        else:
            pass

        # fix systems (make end of previous system beginning of next one)
        for i in xrange(1, un_wrapped_systems.shape[0]):
            un_wrapped_systems[i, 0, 1] = un_wrapped_systems[i - 1, 1, 1]
            un_wrapped_systems[i, 3, 1] = un_wrapped_systems[i - 1, 2, 1]

        return un_wrapped_image, un_wrapped_coords, un_wrapped_systems, un_wrapped_roi_coords
    
    def stitch_image(self, image, systems, notes, system_heigth):
        """
        Re-stitch image
        """
        
        x_offset = 0
        n_notes = 0
        for n in notes:
            n_notes += n.shape[0]
        
        un_wrapped_coords = np.zeros((n_notes, 2))
        
        width = image.shape[1] * systems.shape[0]
        window = int(system_heigth / 2)
        un_wrapped_image = np.zeros((2 * window, width), dtype=np.uint8)
        
        rois = systems_to_rois(systems, window_top=window, window_bottom=window, sort_systems=False)
        
        # make single staff image
        img_start = 0
        notes_start = 0
        for j, r in enumerate(rois):
            # image
            system_image = image[r[0, 0]:r[3, 0], r[0, 1]:r[1, 1]]
            img_end =  img_start + system_image.shape[1]
            un_wrapped_image[:, img_start:img_end] = system_image

            # notes
            unw_notes = notes[j].copy()
            unw_notes[:, 0] -= r[0, 0]
            
            notes_end = notes_start + unw_notes.shape[0]
            un_wrapped_coords[notes_start:notes_end] = unw_notes
            
            img_start = img_end
            notes_start = notes_end
        
        # keep relevant part of unwrapped image
        un_wrapped_image = un_wrapped_image[:, :img_end]
        
#        # clip or pad image in case of scaling
#        if un_wrapped_image.shape[0] > system_heigth:
#            r0 = (un_wrapped_image.shape[0] - system_heigth) // 2
#            r1 = r0 + system_heigth
#            un_wrapped_image = un_wrapped_image[r0:r1, :]
#            un_wrapped_coords[:, 0] -= r0
#        
#        elif un_wrapped_image.shape[0] < system_heigth:
#            missing_top = (system_heigth - un_wrapped_image.shape[0]) // 2
#            missing_bottom = system_heigth - un_wrapped_image.shape[0] - missing_top
#            un_wrapped_image = np.pad(un_wrapped_image, ((missing_top, missing_bottom), (0, 0)), mode='edge')
#            un_wrapped_coords[:, 0] += missing_top
        
#        plt.figure()
#        plt.imshow(un_wrapped_image, cmap=plt.cm.gray)
#        plt.plot(un_wrapped_coords[:, 1], un_wrapped_coords[:, 0], 'co')
#        plt.show(block=True)                
        
        return un_wrapped_image, un_wrapped_coords
    
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
            systems = self.system_pos[i_sheet]
            roi_note_coords = self.roi_note_coords[i_sheet]
            
            spec = self.spectrograms[i_sheet]
            sel_onset = int(self.onsets[i_sheet][i_onset])

            # get sliding window train item
            sliding_window, contained_notes = self.prepare_train_item(sheet, roi_note_coords, i_onset, systems)
            
            # get spectrogram excerpt (target note in center)
            
            # data augmentation note position
            if self.data_augmentation['onset_translation']:
                t = self.data_augmentation['onset_translation']
                sel_onset += np.random.randint(low=-t, high=t+1)
            
            # compute sliding window coordinates
            start = np.max([sel_onset - self.spec_context // 2, 0])
            stop = start + self.spec_context
            
            stop = np.min([stop, spec.shape[1] - 1])
            start = stop - self.spec_context
            
            E = spec[:, start:stop]

            if self.data_augmentation['spec_padding']:
                spec_padding = self.data_augmentation['spec_padding']
                E = np.pad(E, ((spec_padding, spec_padding), (0, 0)), mode='edge')
                s = np.random.randint(0, spec_padding)
                e = s + spec.shape[0]
                E = E[s:e, :]
            
            # compute number of contained onsets
            # contained_onsets = np.sum((self.onsets[i_sheet] > start)  & (self.onsets[i_sheet] < stop))
            
            # self.pair_counts.append([contained_notes, contained_onsets])
            # print np.mean(np.asarray(self.pair_counts), axis=0)
            
            # collect batch data
            Sheet_batch[i_entity, 0, :, :] = sliding_window
            Spec_batch[i_entity, 0, :, :] = E

        return Sheet_batch, Spec_batch


def load_piece_images(piece, version=None):
    """
    Load data for note head training
    """
    import cv2
    from omr.utils.sheet_image import SheetImage

    # initialize sheet images
    sheet_images = []

    # select sheet version
    version = "_%02d" % version if version is not None else ""

    # iterate pieces
    n_coords = 0

    # compile piece directory
    piece_dir = os.path.join(ROOT_DIR, piece)

    # load data
    sheet_dir = os.path.join(piece_dir, "sheet" + version)
    img_files = np.sort(glob.glob(sheet_dir + "/*.*"))

    # check if version exists
    if not os.path.exists(sheet_dir):
        return None

    coord_dir = os.path.join(piece_dir, "coords" + version)
    note_coord_files = np.sort(glob.glob(coord_dir + "/notes_*.npy"))
    system_coord_files = np.sort(glob.glob(coord_dir + "/systems_*.npy"))

    # get number of pages
    n_pages = len(img_files)

    # load data
    for i_page in xrange(n_pages):
        img = cv2.imread(img_files[i_page], 0)

        # load note coords
        note_coords = np.load(note_coord_files[i_page])
        n_coords += len(note_coords)

        # load system coords
        system_coords = np.load(system_coord_files[i_page])

        # initialize sheet image
        sheet_img = SheetImage(img)
        sheet_img.add_annotation("note_coords", note_coords)
        sheet_img.add_annotation("system_coords", system_coords)
        sheet_img.add_annotation("piece", piece)

        sheet_images.append(sheet_img)

    return sheet_images


def systems_to_rois(page_systems, window_top=10, window_bottom=10, sort_systems=True):
    """
    Convert systems to rois
    """

    # sort systems
    if sort_systems:
        sorted_idx = np.argsort(page_systems[:, 0, 0])
        page_systems = page_systems[sorted_idx]

    page_rois = np.zeros((0, 4, 2))
    for s in page_systems:
        cr = np.mean([s[3, 0], s[0, 0]])

        r_min = cr - window_top
        r_max = r_min + window_top + window_bottom
        c_min = s[0, 1]
        c_max = s[1, 1]

        topLeft = [r_min, c_min]
        topRight = [r_min, c_max]
        bottomLeft = [r_max, c_min]
        bottomRight = [r_max, c_max]
        system = np.asarray([topLeft, topRight, bottomRight, bottomLeft])
        system = system.reshape((1, 4, 2))
        page_rois = np.vstack((page_rois, system))

    return page_rois.astype(np.int)


def group_by_roi(coords, systems):
    """
    Group annotations by system
    """
    import matplotlib.path as mplPath

    row_sorted_coords = []
    for system in systems:

        # initialize bounding box
        bbPath = mplPath.Path(system)

        # get notes inside bounding box
        idxs = bbPath.contains_points(coords)

        # sort coordinates of row
        rc = coords[idxs]
        sorted_idx = np.argsort(rc[:, 1])
        rc = rc[sorted_idx]
        row_sorted_coords.append(rc)

    # check if everything went all right
    n_grouped = np.sum([c.shape[0] for c in row_sorted_coords])

    if coords.shape[0] != n_grouped:
        plt.figure("group_by_roi")
        plt.plot(coords[:, 1], coords[:, 0], 'co')

        row_sorted_coords = np.vstack(row_sorted_coords)

        for c in coords:
            if np.sum(c[1] == row_sorted_coords[:, 1]) == 0:
                # print np.sum(c[1] == row_sorted_coords[:, 1])
                plt.plot(c[1], c[0], 'mo', markersize=15)

        # plot rois
        for roi in systems:
            plt.plot([roi[0, 1], roi[0, 1]], [roi[0, 0], roi[3, 0]], '-', color='c')
            plt.plot([roi[1, 1], roi[1, 1]], [roi[1, 0], roi[2, 0]], '-', color='m')

        plt.show(block=True)

    assert coords.shape[0] == n_grouped, "group_by_roi: number of notes changed after grouping (b: %d, a: %d)" % (coords.shape[0], n_grouped)

    return row_sorted_coords


def stack_sheet_images(sheet_images):
    """
    Stack all pages of piece to a single matrix
    """

    y_offset = 0

    stacked_image = np.zeros((0, 835))
    stacked_note_coords = np.zeros((0, 2))
    stacked_system_coords = np.zeros((0, 4, 2))

    # iterate pages
    for i in xrange(len(sheet_images)):
        si = sheet_images[i]
        systems = si.get_annotation("system_coords").copy()
        notes = si.get_annotation("note_coords").copy()

        # sort note heads by system
        window = int(0.9 * SYSTEM_HEIGHT / 2)
        rois = systems_to_rois(systems, window_top=window, window_bottom=window)
        notes = group_by_roi(notes, rois)

        # # debug plot
        # plt.figure()
        # plt.clf()
        # plt.imshow(si.image, cmap=plt.cm.gray)
        # sym = 'o'
        # col = 'm'
        # for j, n in enumerate(notes):
        #     sym = '*' if sym == 'o' else 'o'
        #     col = 'c' if col == 'm' else 'm'
        #     plt.plot(n[:, 1], n[:, 0], sym, color=col, alpha=0.7, markersize=10)
        #     plt.plot([rois[j, 0, 1], rois[j, 1, 1]], [rois[j, 0, 0], rois[j, 1, 0]], '-', color=col)
        #     plt.plot([rois[j, 2, 1], rois[j, 3, 1]], [rois[j, 2, 0], rois[j, 3, 0]], '-', color=col)
        # plt.show(block=True)

        # compile stacked data
        stacked_image = np.vstack((stacked_image, si.image))

        for st_notes in notes:
            st_notes[:, 0] += y_offset
            stacked_note_coords = np.vstack([stacked_note_coords, st_notes])

        systems[:, :, 0] += y_offset
        stacked_system_coords = np.vstack([stacked_system_coords, systems])

        # increase offset
        y_offset += si.image.shape[0]

    # # debug plot (stacked image)
    # plt.figure()
    # plt.imshow(stacked_image, cmap=plt.cm.gray)
    # plt.plot(stacked_note_coords[:, 1], stacked_note_coords[:, 0], 'co')
    # plt.plot(stacked_system_coords[:, :, 1], stacked_system_coords[:, :, 0], 'mo')
    # plt.show(block=True)

    return stacked_image, stacked_note_coords, stacked_system_coords


def prepare_pool(pieces, data_augmentation=None, shuffle=True, one_spec=False, versions=[None],
                 tempi=[], sound_fonts=[]):
    """
    Helper function to prepare train data pool
    """

    images = []
    note_coords = []
    system_pos = []
    specs = []
    onsets = []

    # collect pieces
    for piece in pieces:

        for version in versions:

            # load sheet image with annotations
            sheet_images = load_piece_images(piece, version)

            if sheet_images is None:
                continue

            # load spectrograms and onsets
            spectrogram_files = glob.glob(os.path.join(ROOT_DIR + piece, "spec", piece + "*_spec.npy"))

            for spec_path in spectrogram_files:

                # get tempo and soundfont
                string = spec_path.split('temp_')[1]
                string = string.rsplit('_spec.npy')[0]
                idx = string.find('_')
                tempo = int(string[0:idx])
                sound_font = string[idx+1:]

                # check if data matches requests
                if len(tempi) > 0 and tempo not in tempi:
                    continue

                if len(sound_fonts) > 0 and sound_font not in sound_fonts:
                    continue

                # stack data to one image
                stk_image, stk_note_coords, stk_system_coords = stack_sheet_images(sheet_images)

                # load spectrogram
                spec = np.load(spec_path)
                spec /= spec.max()

                # load onsets
                onset_path = spec_path.replace("_spec.npy", "_onsets.npy")
                cur_onsets = np.load(onset_path)

                # merge onsets occuring in the same frame
                cur_onsets, counts = np.unique(cur_onsets, return_counts=True)
                for i in xrange(len(cur_onsets)):
                    to_delete = slice(i + 1, i + counts[i])
                    stk_note_coords = np.delete(stk_note_coords, to_delete, axis=0)
                    # print cur_onsets.shape, stk_note_coords.shape
                assert len(cur_onsets) == len(stk_note_coords), "number of notes changed after onset merging"

                # collect data
                images.append(stk_image)
                note_coords.append(stk_note_coords)
                system_pos.append(stk_system_coords)

                specs.append(spec)
                onsets.append(cur_onsets)

                if one_spec:
                    break

    # initialize train pool
    pool = AugmentedContinousSpec2SheetHashingPool(sheets=images, coords=note_coords, system_pos=system_pos,
                                                   spectrograms=specs, onsets=onsets,
                                                   spec_context=SPEC_CONTEXT, sheet_context=SHEET_CONTEXT,
                                                   staff_height=SYSTEM_HEIGHT, shuffle=shuffle,
                                                   data_augmentation=data_augmentation)

    return pool


def load_audio_score_retrieval(seed=23, tempi=TR_TEMPI, sound_fonts=TR_SOUNDFONTS):
    """
    Load alignment data
    """
    np.random.seed(seed)

    # compile tag
    tag = "sf_" + "_".join(sound_fonts)
    tag += "_tmp_" + "_".join([str(t) for t in tempi])

    # add agmentation method to tag
    tag += "_strans_" + str(AUGMENT['system_translation'])
    tag += "_sc_" + "_".join([str(s) for s in AUGMENT['sheet_scaling']])
    tag += "_ntrans_" + str(AUGMENT['note_coord_translation'])

    print tag

    # initialize data pools
    train_pool = prepare_pool(tr_pieces, data_augmentation=AUGMENT, versions=[None], tempi=TR_TEMPI, sound_fonts=TR_SOUNDFONTS)
    valid_pool = prepare_pool(va_pieces, data_augmentation=NO_AUGMENT, versions=[None], tempi=TE_TEMPI, sound_fonts=VA_SOUNDFONTS)
    test_pool = prepare_pool(te_pieces, data_augmentation=NO_AUGMENT, versions=[None], tempi=TE_TEMPI, sound_fonts=TE_SOUNDFONTS)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool, train_tag=tag)


def load_realignment_data(seed=23, tempi=TR_TEMPI, sound_fonts=TR_SOUNDFONTS, remove_annotations=True):
    """ Load re-alignment data """
    from realignment_data import compute_alignment

    # load original data
    data = load_audio_score_retrieval(seed=seed, tempi=tempi, sound_fonts=sound_fonts)

    # get train pool
    train_pool = data['train']

    # remove annotation from train pool
    if remove_annotations:
        sheets = train_pool.sheets
        coords = train_pool.coords
        system_pos = train_pool.system_pos
        spectrograms = train_pool.spectrograms
        onsets = train_pool.onsets
        spec_context = train_pool.spec_context
        sheet_context = train_pool.sheet_context
        staff_height = train_pool.staff_height
        data_augmentation = train_pool.data_augmentation

        # iterate pieces
        new_onets = []
        new_coords = []
        new_roi_note_coords = []
        for i in xrange(len(sheets)):
            sheet = sheets[i]

            # compute linear alignment
            o0 = spec_context // 2
            o1 = spectrograms[i].shape[1] - spec_context // 2
            spec_idxs = np.linspace(o0, o1, len(onsets[i])).astype(np.int32)

            c0 = sheet_context // 2
            c1 = sheet.shape[1] - sheet_context // 2
            sheet_idxs = np.linspace(c0, c1, len(spec_idxs)).astype(np.int32)

            img_codes = np.ones((len(onsets[i]), 32))
            spec_codes = np.ones((len(onsets[i]), 32))

            # align data
            a2s_mapping, dtw_res = compute_alignment(img_codes, spec_codes, sheet_idxs, spec_idxs, align_by="baseline")

            # update annotation
            aligne_coords = np.zeros_like(coords[i])
            aligne_coords[:, 1] = dtw_res['aligned_sheet_coords']

            # compute note coords for each system (corresponds to roi)
            aligne_coords[:, 0] = sheet.shape[0] // 2
            roi_note_coords = group_by_roi(aligne_coords, system_pos[i])

            new_onets.append(spec_idxs)
            new_coords.append(aligne_coords)
            new_roi_note_coords.append(roi_note_coords)

        # update annotation
        # -----------------

        train_pool = AugmentedContinousSpec2SheetHashingPool(sheets=sheets, coords=new_coords, system_pos=system_pos,
                                                             spectrograms=spectrograms, onsets=new_onets,
                                                             spec_context=spec_context, sheet_context=sheet_context,
                                                             staff_height=staff_height, shuffle=True,
                                                             data_augmentation=data_augmentation,
                                                             roi_note_coords=new_roi_note_coords)

        # update train pool
        data['train'] = train_pool

    return data


if __name__ == "__main__":
    """ main """

    load_audio_score_retrieval()

    import matplotlib.pyplot as plt

    def train_batch_iterator(batch_size=1):
        """ Compile batch iterator """
        from multi_modality_hashing.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
        batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=None, k_samples=10000)
        return batch_iterator


    # data = load_audio_score_retrieval()
    data = load_realignment_data(remove_annotations=True)
    bi = train_batch_iterator(batch_size=250)
    
    iterator = bi(data["train"])
    
    # show some train samples
    import time
    for epoch in xrange(1000):
        start = time.time()
        for i, (X, Y) in enumerate(iterator):
            print X.shape, Y.shape
            X_va, Y_va = data["valid"][i:i + 1]

            # plt.figure()
            # I = X_va[0, 0]
            #
            # c = int(15. * (200. / 180))
            # In = I[15:-15, c:-c]
            # c = int(5. * (200. / 180))
            # Is = I[10:-10, c:-c]
            # c = int(25. * (200. / 180))
            # Il = I[20:-20, c:-c]
            #
            # import cv2
            # Is = cv2.resize(Is, (200, 180))
            # In = cv2.resize(In, (200, 180))
            # Il = cv2.resize(Il, (200, 180))
            #
            # plt.subplot(1, 3, 1)
            # plt.imshow(Is, cmap=plt.cm.gray)
            # #plt.axis('off')
            #
            # plt.subplot(1, 3, 2)
            # plt.imshow(In, cmap=plt.cm.gray)
            # #plt.axis('off')
            #
            # plt.subplot(1, 3, 3)
            # plt.imshow(Il, cmap=plt.cm.gray)
            # #plt.axis('off')
            #
            # plt.figure()
            # from colormaps import cmaps
            # plt.imshow(Y_va[0, 0], origin='lower', cmap=cmaps['viridis'])
            #
            # plt.show(block=True)
            # continue

            plt.figure("sample image", figsize=(15, 10))
            plt.clf()

            plt.subplot(2, 2, 1)
            plt.imshow(X[0, 0], cmap=plt.cm.gray)
            plt.plot(2 * [SHEET_CONTEXT // 2], [0, SYSTEM_HEIGHT], 'm-')
            plt.colorbar()
            plt.xlabel(SHEET_CONTEXT)
            plt.xlim([0, SHEET_CONTEXT])
            plt.ylim([SYSTEM_HEIGHT, 0])

            plt.subplot(2, 2, 2)
            plt.imshow(Y[0, 0], origin='lower', cmap=plt.cm.jet)
            plt.plot(2 * [SPEC_CONTEXT // 2], [0, SPEC_BINS], 'm-')
            plt.colorbar()
            plt.xlabel(SPEC_CONTEXT)
            plt.ylabel(SPEC_BINS)
            plt.xlim([0, SPEC_CONTEXT - 1])
            plt.ylim([0, SPEC_BINS])

            plt.subplot(2, 2, 3)
            plt.imshow(X_va[0, 0], cmap=plt.cm.gray)
            plt.plot(2 * [SHEET_CONTEXT // 2], [0, SYSTEM_HEIGHT], 'm-')
            plt.colorbar()
            plt.xlabel(SHEET_CONTEXT)
            plt.xlim([0, SHEET_CONTEXT])
            plt.ylim([SYSTEM_HEIGHT, 0])

            plt.subplot(2, 2, 4)
            plt.imshow(Y_va[0, 0], origin='lower', cmap=plt.cm.jet)
            plt.plot(2 * [SPEC_CONTEXT // 2], [0, SPEC_BINS], 'm-')
            plt.colorbar()
            plt.xlabel(SPEC_CONTEXT)
            plt.ylabel(SPEC_BINS)
            plt.xlim([0, SPEC_CONTEXT - 1])
            plt.ylim([0, SPEC_BINS])

            plt.show(block=True)

        stop = time.time()
        print "%.2f seconds required for iterating" % (stop - start)
