
import os
import pickle
import numpy as np
from multi_modality_hashing.config.settings import DATA_ROOT_FLICKR30k, DATA_ROOT_A2S, DATA_ROOT_FREESOUND,\
    DATA_ROOT_FLICKR8k, DATA_ROOT_IAPR


class ContinousSpec2SheetHashingPool(object):
    """
    Data Pool for spectrogram to sheet snippet hashing
    """

    def __init__(self, sheets, x_coords, start_pos, spectrograms, onsets, spec_context, sheet_context, staff_height=40, shuffle=True):
        """
        Constructor
        """

        self.sheets = sheets
        self.x_coords = x_coords
        self.start_pos = start_pos
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

            # iterate onsets in sheet
            for i_onset in xrange(0, len(self.onsets[i_sheet])):
                onset = self.onsets[i_sheet][i_onset]
                if (onset - self.spec_context) >= 0 and onset < spec.shape[1]:
                    cur_entities = np.asarray([i_sheet, i_onset])
                    self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self, indices=None):
        """ reset batch generator """
        if indices is None:
            indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def prepare_train_item(self, sheet, coords, i_onset, start_pos):
        """ prepare train item (sliding window + new target coords) """

        # target coordinate
        x = int(coords[i_onset, 1])

        # compute sliding window coordinates
        x0 = x - self.sheet_context
        x0_end = x

        # find concerned row
        dists = np.abs(coords[i_onset, 0] - start_pos[:, 0])
        i_row = np.argmin(dists)

        if x0 >= 0:
            i_row0 = i_row
            x1_start = x1 = 0
        else:
            i_row0 = i_row - 1
            x0 += sheet.shape[1]
            x0_end = sheet.shape[1]
            x1_start = 0
            x1 = self.sheet_context - (sheet.shape[1] - x0)

        y0 = int(start_pos[i_row0, 0] - (self.staff_height // 2))
        y1 = int(start_pos[i_row0, 0] + (self.staff_height // 2))
        I0 = sheet[y0:y1, x0:x0_end]

        y0 = int(start_pos[i_row, 0] - (self.staff_height // 2))
        y1 = int(start_pos[i_row, 0] + (self.staff_height // 2))
        I1 = sheet[y0:y1, x1_start:x1]

        sliding_window = np.concatenate((I0, I1), axis=1)

        return sliding_window

    def __getitem__(self, key):
        """ make class accessible by index or slice """

        # get batch
        if key.__class__ != slice:
            key = slice(key, key + 1)
        batch_entities = self.train_entities[key]

        # collect train entities
        Sheet_batch = np.zeros((len(batch_entities), 1, self.sheet_dim[0], self.sheet_context), dtype=np.float32)
        Spec_batch = np.zeros((len(batch_entities), 1, self.spec_dim[0], self.spec_context), dtype=np.float32)
        x_scores = np.zeros((len(batch_entities), 1), dtype=np.float32)
        for i_entity, (i_sheet, i_onset) in enumerate(batch_entities):

            # get sheet and annotations
            sheet = self.sheets[i_sheet]
            start_pos = self.start_pos[i_sheet]
            spec = self.spectrograms[i_sheet]
            sel_onset = int(self.onsets[i_sheet][i_onset])
            coords = self.x_coords[i_sheet]

            # get sliding window train item
            sliding_window = self.prepare_train_item(sheet, coords, i_onset, start_pos)

            # --- debug plot --------------------------------------------------
            # plt.figure("Sheet", figsize=(19, 10))
            # plt.clf()
            # plt.subplot(1, 2, 1)
            # plt.imshow(1.0 - sheet, cmap=plt.cm.gray, interpolation='nearest')
            # plt.ylim([sheet.shape[0] - 1, 0])
            # plt.xlim([0, sheet.shape[1] - 1])
            # plt.axis('off')
            #
            # # highlight target note
            # plt.plot(coords[i_onset, 1], coords[i_onset, 0], 'mo')
            #
            # # plot start positions
            # plt.plot(start_pos[:, 1], start_pos[:, 0], 'mo', markersize=10)
            #
            # plt.subplot(1, 2, 2)
            # plt.imshow(1.0 - sliding_window, cmap=plt.cm.gray, interpolation='nearest')
            # plt.ylim([sliding_window.shape[0]-1, 0])
            # plt.xlim([0, sliding_window.shape[1]-1])
            # plt.axis('off')
            #
            # plt.plot(target_coord[1], target_coord[0], 'co', markersize=10)
            #
            # plt.show(block=True)
            # -----------------------------------------------------------------

            E = spec[:, sel_onset - self.spec_context:sel_onset]

            # collect batch data
            Sheet_batch[i_entity, 0, :, :] = sliding_window
            Spec_batch[i_entity, 0, :, :] = E

        return Sheet_batch, Spec_batch


class Image2CaptionPool(object):
    """
    Data pool for hashing captions to images
    """

    def __init__(self, images, captions, shuffle=True):
        """
        Constructor
        """

        self.images = images
        self.captions = captions

        self.img_dim = list(self.images[0].shape)
        self.cap_dim = list(captions[0][0].shape)

        self.shape = None

        # prepare train data
        self.train_entities = None
        self.prepare_train_entities()

        # shuffle data
        if shuffle:
            self.reset_batch_generator()

    def prepare_train_entities(self):
        """ collect train entities """

        self.train_entities = np.zeros((0, 2), dtype=np.int)

        # iterate images and captions
        for i_img in xrange(self.images.shape[0]):
            for i_caption in xrange(self.captions[i_img].shape[0]):
                cur_entities = np.asarray([i_img, i_caption])
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
        if key.__class__ != slice:
            key = slice(key, key + 1)
        batch_entities = self.train_entities[key]

        # collect train entities
        img_batch = np.zeros(tuple([len(batch_entities)] + self.img_dim), dtype=np.float32)
        cap_batch = np.zeros(tuple([len(batch_entities)] + self.cap_dim), dtype=np.float32)
        for i_entity, (i_img, i_cap) in enumerate(batch_entities):
            # get sheet and annotations
            img = self.images[i_img]
            cap = self.captions[i_img][i_cap]

            # collect batch data
            img_batch[i_entity] = img
            cap_batch[i_entity] = cap

        return img_batch, cap_batch


class SubSequenceDataPool(object):
    """ Data Pool for Sub-Sequences of Sequence Collection """

    def __init__(self, sequences, vocabulary, seq_len=10, step_size=1, seed=0):
        """
        Constructor

        :param sequences: list of one or more sequences
        :param targets: list of target sequences (corresponding to sequence list)
        :param seq_len: length of sub-sequences to be returned
        :param step_size: step size for sub-sequence sliding window
        """

        self.sequences = sequences
        self.seq_len = seq_len
        self.step_size = step_size
        self.vocabulary = vocabulary

        # seed for random permutations (required for two or more synchronized SubSequenceDataPools)
        self.seed = seed

        # number of sub-sequences
        self.shape = None
        self.train_entities = None

        # get dimensions of input sequence
        self.ndim = len(self.vocabulary.keys())

        # prepare data
        self.prepare_train_entities()

        # shuffle data
        self.reset_batch_generator()

    def prepare_train_entities(self):
        """
        Prepare train entities

        A train entity is a tuple such as (sequence_idx, sub_sequence_start_idx)
        """
        self.train_entities = np.zeros((0, 2), dtype=np.int)

        # iterate all sequences
        for i, sequence in enumerate(self.sequences):

            # load sequence if file path is given
            if sequence.__class__ == np.string_:
                sequence = np.load(sequence)

            # compile current train entities
            tr_indices = np.arange(0, sequence.shape[0] - self.seq_len, self.step_size).astype(np.int)
            cur_entities = np.asarray(zip(np.repeat(i, len(tr_indices)), tr_indices), dtype=np.int)

            self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self, indices=None):
        """ Reset batch generator (random shuffle train data) """
        if indices is None:
            np.random.seed(self.seed)
            indices = np.random.permutation(self.shape[0])
            self.seed += 1
        self.train_entities = self.train_entities[indices]

    def __getitem__(self, key):
        """ Make class accessible by index or slice """

        # get batch
        if key.__class__ != slice:
            key = slice(key, key+1)
        batch_entities = self.train_entities[key]

        # initialize batch data
        X_batch = np.zeros((len(batch_entities), self.seq_len, self.ndim), dtype=np.float32)

        # collect batch data
        for i_batch_entity, (i_sequence, t) in enumerate(batch_entities):

            # load sequence if necessary
            sequence = self.sequences[i_sequence]
            if sequence.__class__ == np.string_:
                sequence = np.load(sequence)

            # collect sub-sequence
            X_batch[i_batch_entity, :, :] = sequence[slice(t, t + self.seq_len)]

        return X_batch

    def __len__(self):
        return self.shape[0]


def load_bucket_spec2score_continous(spec_context=100, sheet_context=100, small_set=False):
    """
    Load alignment data
    """
    np.random.seed(23)

    dset = 'Nottingham'
    dump_file_tr = DATA_ROOT_A2S + ('/%s/train_full_sheet_data.pkl' % dset)
    dump_file_va = DATA_ROOT_A2S + ('/%s/valid_full_sheet_data.pkl' % dset)
    dump_file_te = DATA_ROOT_A2S + ('/%s/test_full_sheet_data.pkl' % dset)

    with open(dump_file_tr, 'rb') as fp:
        tr_data = pickle.load(fp)

    with open(dump_file_va, 'rb') as fp:
        va_data = pickle.load(fp)

    with open(dump_file_te, 'rb') as fp:
        te_data = pickle.load(fp)

    def make_pool(sel_data):
        """ make data pool """

        sheets = [entry[0] for entry in sel_data]
        start_pos = [entry[1] for entry in sel_data]
        x_coords = [entry[2] for entry in sel_data]
        spectrograms = [entry[3] for entry in sel_data]
        onsets = [entry[4] for entry in sel_data]

        return ContinousSpec2SheetHashingPool(sheets, x_coords, start_pos, spectrograms, onsets, spec_context, sheet_context)

    # ------------------------------------------------------------------------

    # initialize data pools
    if small_set == "13k":  # 10%
        train_pool = make_pool(tr_data[:21])
    elif small_set == "27k":  # 27%
        train_pool = make_pool(tr_data[:47])
    elif small_set == "68k":  # 25%
        train_pool = make_pool(tr_data[:110])
    elif small_set == "135k":  # 50%
        train_pool = make_pool(tr_data[:207])
    else:
        train_pool = make_pool(tr_data[:])
    valid_pool = make_pool(va_data[:])
    test_pool = make_pool(te_data[:])

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_flickr30k_rnn(img_file="images_vgg_vectors.npy", cap_file="captions_onehot.pkl", normalize_img=True, seed=23):
    """
    Load Flickr30k data set
    """
    np.random.seed(seed)

    # load images
    Images = np.load(DATA_ROOT_FLICKR30k + img_file)

    # load caption vectors
    with open(DATA_ROOT_FLICKR30k + cap_file, "rb") as fp:
        vector_captions, captions, img_ids = pickle.load(fp)

    # normalize images
    print "Images.max()", Images.max()
    if normalize_img:
        Images = Images.astype(np.float32)
        Images /= Images.max()

    tr_images = Images[0:28000]
    va_images = Images[-1000:]
    te_images = Images[-2000:-1000]

    tr_captions = vector_captions[0:28000]
    va_captions = vector_captions[-1000:]
    te_captions = vector_captions[-2000:-1000]

    # initialize data pools
    train_pool = Image2CaptionPool(tr_images, tr_captions)
    valid_pool = Image2CaptionPool(va_images, va_captions)
    test_pool  = Image2CaptionPool(te_images, te_captions)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_kiros(name='f30k', seed=23):
    """
    Load Flickr30k data set
    """

    def build_dictionary(text):
        """
        Build a dictionary
        text: list of sentences (pre-tokenized)
        """
        from collections import OrderedDict
        wordcount = OrderedDict()
        for cc in text:
            words = cc.split()
            for w in words:
                if w not in wordcount:
                    wordcount[w] = 0
                wordcount[w] += 1
        words = wordcount.keys()
        freqs = wordcount.values()
        sorted_idx = np.argsort(freqs)[::-1]

        worddict = OrderedDict()
        for idx, sidx in enumerate(sorted_idx):
            worddict[words[sidx]] = idx + 2  # 0: <eos>, 1: <unk>

        return worddict, wordcount

    np.random.seed(seed)
    load_train = True
    max_words = 100

    path_to_data = '/home/matthias/cp/src/visual-semantic-embedding/data/'
    loc = path_to_data + name + '/'

    # Captions
    train_caps, dev_caps, test_caps = [], [], []
    if load_train:
        with open(loc + name + '_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())
    else:
        train_caps = None
    with open(loc + name + '_dev_caps.txt', 'rb') as f:
        for line in f:
            dev_caps.append(line.strip())
    with open(loc + name + '_test_caps.txt', 'rb') as f:
        for line in f:
            test_caps.append(line.strip())

    # Image features
    if load_train:
        train_ims = np.load(loc + name + '_train_ims.npy')
    else:
        train_ims = None

    dev_ims = np.load(loc + name + '_dev_ims.npy')
    test_ims = np.load(loc + name + '_test_ims.npy')

    tr_images = train_ims
    va_images = dev_ims
    te_images = test_ims

    tr_captions = train_caps
    va_captions = dev_caps
    te_captions = test_caps

    # compute dictionary
    worddict = build_dictionary(tr_captions + va_captions)[0]

    # compute vector encodings of captions
    def vectorize_captions(captions):
        vectorized_captions = -np.ones((len(captions), 1, max_words), dtype=np.int32)
        for i_cap, caption in enumerate(captions):

            for i_word, w in enumerate(caption.split()):
                w_idx = worddict[w] if worddict[w] < max_words else 1
                vectorized_captions[i_cap, 0, i_word] = w_idx

        return vectorized_captions

    tr_captions = vectorize_captions(tr_captions)
    va_captions = vectorize_captions(va_captions)
    # te_captions = vectorize_captions(te_captions)

    va_captions = va_captions[0:5000:5]
    va_images = va_images[0:5000:5]

    te_images = va_images
    te_captions = va_captions

    # initialize data pools
    train_pool = Image2CaptionPool(tr_images, tr_captions)
    valid_pool = Image2CaptionPool(va_images, va_captions)
    test_pool = Image2CaptionPool(te_images, te_captions)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_flickr30k(img_file="images.npy", cap_file="captions.pkl", normalize_img=True, normalize_txt=True, seed=23):
    """
    Load Flickr30k data set
    """
    np.random.seed(seed)

    # load images
    Images = np.load(DATA_ROOT_FLICKR30k + img_file)

    # load caption vectors
    with open(DATA_ROOT_FLICKR30k + cap_file, "rb") as fp:
        vector_captions, captions, img_ids = pickle.load(fp)

    # normalize images
    print "Images.max()", Images.max()
    if normalize_img:
        Images = Images.astype(np.float32)
        Images /= Images.max()

    # normalize vector space
    if normalize_txt:
        vector_captions -= vector_captions.min()
        vector_captions /= vector_captions.max()

    tr_images = Images[0:28000]
    va_images = Images[-1000:]
    te_images = Images[-2000:-1000]

    tr_captions = vector_captions[0:28000]
    va_captions = vector_captions[-1000:]
    te_captions = vector_captions[-2000:-1000]

    # initialize data pools
    train_pool = Image2CaptionPool(tr_images, tr_captions)
    valid_pool = Image2CaptionPool(va_images, va_captions, shuffle=True)
    test_pool  = Image2CaptionPool(te_images, te_captions, shuffle=True)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_flickr30k_II(img_file="images.npy", cap_file="captions_tfidf_unpooled.pkl", normalize_img=True, normalize_txt=True, seed=23):
    """
    Load Flickr30k data set
    """
    np.random.seed(seed)

    # load images
    Images = np.load(DATA_ROOT_FLICKR30k + img_file)

    # load caption vectors
    with open(DATA_ROOT_FLICKR30k + cap_file, "rb") as fp:
        caption_vectors_tr, caption_vectors_va, captions, img_ids = pickle.load(fp)

    # normalize images
    if normalize_img:
        Images = Images.astype(np.float32)
        Images /= Images.max()

    # normalize vector space
    if normalize_txt:
        caption_vectors_va -= caption_vectors_tr.min()
        caption_vectors_tr -= caption_vectors_tr.min()
        caption_vectors_va /= caption_vectors_tr.max()
        caption_vectors_tr /= caption_vectors_tr.max()

    tr_images = Images[0:28000]
    va_images = Images[-1000:]
    te_images = Images[-2000:-1000]

    tr_captions = caption_vectors_tr[0:28000]
    va_captions = caption_vectors_va[-1000:]
    te_captions = caption_vectors_va[-2000:-1000]

    # initialize data pools
    train_pool = Image2CaptionPool(tr_images, tr_captions)
    valid_pool = Image2CaptionPool(va_images, va_captions, shuffle=False)
    test_pool  = Image2CaptionPool(te_images, te_captions, shuffle=False)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_flickr8k(img_file="images_vgg_vectors.npy", cap_file="captions_tfidf.pkl",
                  normalize_img=True, normalize_txt=True, seed=23):
    """
    Load Flickr8k data set
    """
    np.random.seed(seed)
    
    def get_set_indices(img_ids, set_file):
        tr_set_file = os.path.join(DATA_ROOT_FLICKR8k, "Flickr8k_text", set_file)
        with open(tr_set_file, 'rb') as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]
        tr_ids = np.asarray(lines)
        return np.in1d(img_ids, tr_ids)

    # load images
    Images = np.load(DATA_ROOT_FLICKR8k + img_file)

    # load caption vectors
    with open(DATA_ROOT_FLICKR8k + cap_file, "rb") as fp:
        vector_captions, captions, img_ids = pickle.load(fp)
    
    # normalize images
    print "Images.max()", Images.max()
    if normalize_img:
        Images = Images.astype(np.float32)
        Images /= Images.max()

    # normalize vector space
    if normalize_txt:
        vector_captions -= vector_captions.min()
        vector_captions /= vector_captions.max()

    # img ids to list
    img_ids = np.asarray(img_ids)
    
    # stack image ids for fliped image vectors
    fliped_img_ids = np.concatenate((img_ids, img_ids))
    vector_captions = np.concatenate((vector_captions, vector_captions))
    
    # load splits
    tr_idxs = get_set_indices(fliped_img_ids, "Flickr_8k.trainImages.txt")
    va_idxs = get_set_indices(img_ids, "Flickr_8k.devImages.txt")
    te_idxs = get_set_indices(img_ids, "Flickr_8k.testImages.txt")

    va_idxs = np.concatenate((va_idxs, np.zeros_like(va_idxs)))
    te_idxs = np.concatenate((te_idxs, np.zeros_like(te_idxs)))

    tr_images = Images[tr_idxs]
    va_images = Images[va_idxs]
    te_images = Images[te_idxs]

    tr_captions = vector_captions[tr_idxs]
    va_captions = vector_captions[va_idxs]
    te_captions = vector_captions[te_idxs]

    # initialize data pools
    train_pool = Image2CaptionPool(tr_images, tr_captions)
    valid_pool = Image2CaptionPool(va_images, va_captions)
    test_pool  = Image2CaptionPool(te_images, te_captions)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_iapr(img_file="images_vgg_vectors.npy", cap_file="captions_tfidf.pkl", normalize_img=True, normalize_txt=True, seed=23):
    """
    Load Flickr30k data set
    """
    np.random.seed(seed)

    # load images
    Images = np.load(DATA_ROOT_IAPR + img_file)

    # load caption vectors
    with open(DATA_ROOT_IAPR + cap_file, "rb") as fp:
        vector_captions, captions = pickle.load(fp)

    # normalize images
    print "Images.max()", Images.max()
    if normalize_img:
        Images = Images.astype(np.float32)
        Images /= Images.max()

    # normalize vector space
    if normalize_txt:
        vector_captions -= vector_captions.min()
        vector_captions /= vector_captions.max()

    # split images into original and flipped version
    Images2 = Images[19996:]
    Images = Images[0:19996]

    # shuffle the data
    rand_idx = np.random.permutation(len(vector_captions))
    Images = Images[rand_idx]
    Images2 = Images2[rand_idx]
    vector_captions = vector_captions[rand_idx]

    #tr_images = np.concatenate([Images[3000:], Images2[3000:]])
    tr_images = Images[3000:]
    va_images = Images[0:1000]
    te_images = Images[1000:3000]

    #tr_captions = np.concatenate([vector_captions[3000:], vector_captions[3000:]])
    tr_captions = vector_captions[3000:]
    va_captions = vector_captions[0:1000]
    te_captions = vector_captions[1000:3000]

    # reuse some images to have 17000 samples available
    tr_images = np.concatenate([tr_images, tr_images[0:4]])
    tr_captions = np.concatenate([tr_captions, tr_captions[0:4]])

    # initialize data pools
    train_pool = Image2CaptionPool(tr_images, tr_captions)
    valid_pool = Image2CaptionPool(va_images, va_captions, shuffle=False)
    test_pool  = Image2CaptionPool(te_images, te_captions, shuffle=False)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_word_rnn(min_word_count=5):
    """ Load data for word prediction rnn """
    from collections import Counter
    
    # load data
    key2sentence = pickle.load(open(DATA_ROOT_FLICKR30k + "caption_dict.pkl", 'rb'))
    
    max_length = 0
    min_length = 1e9
    
    # count words
    counter = Counter()
    for c in key2sentence.values():
        counter.update(c)
        max_length = np.max([max_length, len(c)])
        min_length = np.min([min_length, len(c)])
    print("Total words:", len(counter))
    print("Minimum caption length:", min_length)
    print("Maximum caption length:", max_length)
    
    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    
    # add unknown word
    vocab_dict["<UNK>"] = len(word_counts)
    vocab_size = len(vocab_dict.keys())
    
    # convert captions to one hot word encodings
    n_captions = 2000
    X = np.zeros((n_captions, 10, vocab_size), dtype=np.int32)
    M = np.zeros((n_captions, 10), dtype=np.int32)
    for i, c in enumerate(key2sentence.values()):
        
        if i >= n_captions:
            break        
        
        for t, w in enumerate(c):

            if t >= 10:
                break
            
            if w in vocab_dict:
                idx = vocab_dict[w]
            else:
                idx = vocab_dict["<UNK>"]
            
            X[i, t, idx] = 1
            M[i, 0:idx] = 1
        
    X_tr = X[0:1500]
    X_va = X[1500::]
    
    from SequenceDataPool import LastStepPredictionDataPool
    train = LastStepPredictionDataPool(X_tr, max_len=10, y_train=None, shuffle=True)
    valid = LastStepPredictionDataPool(X_va, max_len=10, y_train=None, shuffle=True)
    test = LastStepPredictionDataPool(X_va, max_len=10, y_train=None, shuffle=True)
    
    data = dict()
    data['train'] = train
    data['valid'] = valid
    data['test'] = test
    
    return data, 10


def load_freesound():
    """ Load freesound samples """
    
    tag_path = "/media/matthias/Data/freesound/freesound_tags_features/tags.dat"
    tags = open(tag_path, "r").read().split("\n")[0:-1]
    
    id_path = "/media/matthias/Data/freesound/freesound_tags_features/ids.dat"
    ids = open(id_path, "r").read().split("\n")[0:-1]
        
    features_path = "/media/matthias/Data/freesound/freesound_tags_features/features.dat"    
    features = open(features_path, "r").read().split("\n")[0:-1]
    features = [np.asarray(f.split(','), dtype=np.float32) for f in features]
    features = np.asarray(features, dtype=np.float32)
    
    with open(DATA_ROOT_FREESOUND + "tags_tfidf.pkl", "rb") as fp:
        tag_vectors = pickle.load(fp)
    
    features = features[0:len(tag_vectors)]

    # init normalization data
    norm_data = dict()

    # normalize data
    norm_data["mfcc"] = {"max": features.max()}
    features /= norm_data["mfcc"]["max"]

    # normalize vector space
    norm_data["text"] = {"min": tag_vectors.min()}
    tag_vectors -= norm_data["text"]["min"]
    norm_data["text"]["max"] = tag_vectors.max()
    tag_vectors /= norm_data["text"]["max"]

    # dump normalization data
    with open(DATA_ROOT_FREESOUND + 'data_normalization.pkl', 'wb') as fp:
        pickle.dump(norm_data, fp)

    # shuffle data
    np.random.seed(23)
    rand_idx = np.random.permutation(len(features))
    features = features[rand_idx]
    tag_vectors = tag_vectors[rand_idx]
    
    n_tr, n_va, n_te = 73000, 1000, 1000
    
    tr_features = features[0:n_tr]
    va_features = features[-(n_va + n_te):-n_te]
    te_features = features[-n_te::]

    tr_tags = tag_vectors[0:n_tr]
    va_tags = tag_vectors[-(n_va + n_te):-n_te]
    te_tags = tag_vectors[-n_te::]

    # initialize data pools
    train_pool = Image2CaptionPool(tr_features, tr_tags)
    valid_pool = Image2CaptionPool(va_features, va_tags)
    test_pool  = Image2CaptionPool(te_features, te_tags)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])
    
    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_multi_view_mnist(small_set=False):
    """
    Load multi view mnist data set
    """
    np.random.seed(seed=0)
    import gzip
    import cPickle
    from scipy import ndimage

    with gzip.open('/home/matthias/cp/data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    # load labels
    y_train = np.asarray(train_set[1], dtype=np.int32)
    y_valid = np.asarray(valid_set[1], dtype=np.int32)
    y_test = np.asarray(test_set[1], dtype=np.int32)

    # load images
    X_tr, X_va, X_te = train_set[0], valid_set[0], test_set[0]

    # remove some data for faster training
    if small_set:
        X_tr, y_train = X_tr[0:10000], y_train[0:10000]

    def prepare_set(X, y, factor=1):
        """ Prepare noisy mulit view train set """
        V1 = np.zeros([X.shape[0] * factor, 1, 28, 28], dtype=np.float32)
        V2 = np.zeros([X.shape[0] * factor, 1, 28, 28], dtype=np.float32)
        y_out = np.zeros(y.shape[0] * factor, dtype=np.int32)

        for iteration in xrange(factor):
            for i in xrange(X.shape[0]):
                idx = iteration * X.shape[0] + i

                # get view 1 image
                View_1, y1 = X[i], y[i]

                # get view 2 image
                cand_idxs = np.nonzero(y == y1)[0]
                j = cand_idxs[np.random.randint(0, cand_idxs.shape[0])]
                View_2 = X[j]

                # reshape to images
                View_1 = View_1.reshape((28, 28))
                View_2 = View_2.reshape((28, 28))

                # rotate view 1
                angle = np.random.uniform(-45, 45)
                View_1 = ndimage.rotate(View_1, angle, reshape=False, order=1)

                # add noise to view 2
                View_2 += 0.3 * np.random.uniform(size=View_2.shape)
                View_2[View_2 > 1] = 1

                # V1[idx] = View_1.reshape((28*28))
                # V2[idx] = View_2.reshape((28*28))
                V1[idx, 0] = View_1
                V2[idx, 0] = View_2

                # get label
                y_out[idx] = y1

                #                plt.figure("View 1 - View 2")
                #                plt.clf()
                #                plt.subplot(1, 2, 1)
                #                plt.imshow(View_1, interpolation='nearest', cmap=plt.cm.gray)
                #                plt.title(y1)
                #                plt.subplot(1, 2, 2)
                #                plt.imshow(View_2, interpolation='nearest', cmap=plt.cm.gray)
                #                plt.show(block=True)

        return V1, V2, y_out

    V1_tr, V2_tr, y_train = prepare_set(X_tr, y_train, factor=1)
    V1_va, V2_va, y_valid = prepare_set(X_va, y_valid)
    V1_te, V2_te, y_test = prepare_set(X_te, y_test)

    print " #Train Samples:", V1_tr.shape
    print " #Validation Samples:", V1_va.shape
    print " #Test Samples:", V1_te.shape

    return dict(X_train=V1_tr, y_train=V2_tr, target_train=y_train,
                X_valid=V1_va, y_valid=V2_va, target_valid=y_valid,
                X_test=V1_te, y_test=V2_te, target_test=y_test)


if __name__ == "__main__":
    """ main """

    # data = load_flickr30k_rnn()

    data = load_kiros()

    # data = load_flickr30k_rnn(img_file="images_vgg_vectors.npy", cap_file="captions_onehot.pkl", normalize_img=True, seed=23)
    # X, Y = data['train'][0]
    # print X.shape
    # print Y.shape
    #
    # global lengths
    # lengths = []
    #
    # def prepare(x, y):
    #     """ Prepare one hot vector text encodings for rnn training """
    #     import numpy as np
    #     y_one_hot = np.zeros((y.shape[0], N_STEPS, ONE_HOT_DIM), dtype=np.float32)
    #
    #     for i in xrange(y.shape[0]):
    #         y_sub = y[i].astype(np.int32)
    #         y_sub = y_sub[y_sub >= 0]
    #
    #         seq_len = len(y_sub)
    #
    #         global lengths
    #         lengths.append(seq_len)
    #
    #         # sequence is longer than network input
    #         if seq_len >= N_STEPS:
    #             s = np.random.randint(0, seq_len - N_STEPS + 1)
    #             y_sub = y_sub[s:s + N_STEPS]
    #
    #         # sequence is shorter than network input
    #         else:
    #             while len(y_sub) < N_STEPS:
    #                 y_sub = np.concatenate((y_sub, y_sub))
    #             y_sub = y_sub[-N_STEPS::]
    #
    #         for j in xrange(N_STEPS):
    #             y_one_hot[i, j, y_sub[j]] = 1
    #
    #     return x, y_one_hot
    #
    # from multi_modality_hashing.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    # batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=1000, k_samples=28000, prepare=prepare)
    #
    # N_STEPS = 60
    # ONE_HOT_DIM = 6184
    #
    # for V1_1, V2_1 in batch_iterator(data['train']):
    #     print V1_1.shape, V2_1.shape
    #
    # print "min_len", np.min(lengths)
    # print "med_len", np.median(lengths)

    # import matplotlib.pyplot as plt
    # from multi_modality_hashing.utils.colormaps import cmaps
    #
    # data = load_flickr30k(img_file="images_vgg_vectors.npy", cap_file="captions_tfidf.pkl")
    #
    # from batch_iterators import MultiviewPoolIteratorUnsupervised
    # batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=500, prepare=None, k_samples=28000)
    # for e in xrange(2):
    #     X1 = []
    #     for V1_1, V2_1 in batch_iterator(data['train']):
    #         print V1_1.shape, V2_1.shape
    #         X1.append(V1_1)
    #
    # batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=500, prepare=None, k_samples=7000)
    # for e in xrange(2):
    #     X2 = []
    #     for _ in xrange(4):
    #         for V1_2, V2_2 in batch_iterator(data['train']):
    #             X2.append(V1_2)
    #
    # X1 = np.asarray(X1)
    # X2 = np.asarray(X2)
    # print X1.shape, X2.shape
    # print np.allclose(X1, X2)

    data = load_bucket_spec2score_continous(spec_context=100, sheet_context=100)
    print "Done!"

    import matplotlib.pyplot as plt
    from batch_iterators import MultiviewPoolIteratorUnsupervised
    batch_iterator = MultiviewPoolIteratorUnsupervised(4)
    for Sheet_batch, Spec_batch in batch_iterator(data['train']):
        print Sheet_batch.shape, Spec_batch.shape

        plt.figure("Samples")
        plt.clf()
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, wspace=0.05, hspace=0.02)

        for i in xrange(4):

            plt.subplot(2, 4, i + 1)
            plt.imshow(1.0 - Sheet_batch[i, 0], cmap=plt.cm.gray)
            plt.axis('off')

            plt.subplot(2, 4, i + 5)
            plt.imshow(Spec_batch[i, 0], origin='lower', aspect='auto')
            plt.axis('off')

        plt.show(block=True)
