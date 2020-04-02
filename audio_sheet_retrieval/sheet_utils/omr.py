
from __future__ import print_function

import cv2
import pickle
import theano
import lasagne
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

from audio_sheet_retrieval.utils.net_utils import load_net_params


def prepare_image(img):
    img = img.astype(np.float32)
    if img.max() != 0:
        img /= img.max()
    return img


def snap_system_to_grid(image, min_row, max_row, min_col, max_col):
    """ snap system to line grid"""

    # pre-process image
    image = 1.0 - image[0, 0]
    imagex = cv2.blur(image, (1, 3))
    imagey = cv2.blur(image, (3, 1))

    # y-direction
    edge_signal = imagey.mean(axis=1)
    edge_candidates = peak_local_max(edge_signal, threshold_rel=0.5)

    # plt.figure("Edges")
    # plt.subplot(2, 1, 1)
    # plt.imshow(imagey)
    # plt.subplot(2, 1, 2)
    # plt.plot(edge_signal, '-')
    # plt.plot(edge_candidates, edge_signal[edge_candidates], 'o')
    # plt.xlim([0, len(edge_signal)])
    # plt.show(block=True)

    min_dists = np.abs(min_row - edge_candidates)
    min_idx_min = np.argmin(min_dists)
    min_dist = min_dists[min_idx_min]

    max_dists = np.abs(max_row - edge_candidates)
    min_idx_max = np.argmin(max_dists)
    max_dist = max_dists[min_idx_max]

    # print min_idx_min, min_idx_max
    # print min_dist, max_dist
    # print "-" * 50

    # update system if candidate found
    thresh = 10
    if min_dist < thresh and max_dist < thresh:
        min_row = edge_candidates[min_idx_min, 0]
        max_row = edge_candidates[min_idx_max, 0]

    # x-direction
    edge_signal = imagex[min_row:max_row, :].mean(axis=0)
    edge_candidates = peak_local_max(edge_signal, threshold_rel=0.5)

    # plt.figure("Edges")
    # plt.subplot(2, 1, 1)
    # plt.imshow(imagex[min_row:max_row, :])
    # plt.subplot(2, 1, 2)
    # plt.plot(edge_signal, '-')
    # plt.plot(edge_candidates, edge_signal[edge_candidates], 'o')
    # plt.xlim([0, len(edge_signal)])
    # plt.show(block=True)

    min_dists = np.abs(min_col - edge_candidates)
    min_idx_min = np.argmin(min_dists)
    min_dist = min_dists[min_idx_min]

    max_dists = np.abs(max_row - edge_candidates)
    min_idx_max = np.argmin(max_dists)
    max_dist = max_dists[min_idx_max]

    # print min_idx_min, min_idx_max
    # print min_dist, max_dist
    # print "-" * 50

    # update system if candidate found
    thresh = 10
    if min_dist < thresh and max_dist < thresh:
        min_col = edge_candidates[min_idx_min, 0]
        max_col = edge_candidates[min_idx_max, 0]

    return min_row, max_row, min_col, max_col


class Network(object):
    """
    Neural Network
    """

    def __init__(self, net):
        """
        Constructor
        """
        self.net = net
        self.compute_output = None
        self.compute_output_dict = dict()
        self.saliency_function = None

        # get input shape of network
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        self.input_shape = l_in.output_shape

    def predict_proba(self, input):
        """
        Predict on test samples
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()

        return self.compute_output(*input)

    def predict(self, input):
        """
        Predict class labels on test samples
        """
        return np.argmax(self.predict_proba(input), axis=1)

    def compute_layer_output(self, input, layer):
        """
        Compute output of given layer
        layer: either a string (name of layer) or a layer object
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        # get layer by name
        if not isinstance(layer, lasagne.layers.Layer):
            for l in lasagne.layers.helper.get_all_layers(self.net):
                if l.name == layer:
                    layer = l
                    break

        # compile prediction function for target layer
        if layer not in self.compute_output_dict:
            self.compute_output_dict[layer] = self._compile_prediction_function(target_layer=layer)

        return self.compute_output_dict[layer](*input)

    def save(self, file_path):
        """
        Save model to disk
        """
        with open(file_path, 'w') as fp:
            params = lasagne.layers.get_all_param_values(self.net)
            pickle.dump(params, fp, -1)

    def load(self, file_path):
        """
        load model from disk
        """
        params = load_net_params(file_path)
        lasagne.layers.set_all_param_values(self.net, params)

    def _compile_prediction_function(self, target_layer=None):
        """
        Compile theano prediction function
        """

        # get network output nad compile function
        if target_layer is None:
            target_layer = self.net

        # collect input vars
        all_layers = lasagne.layers.helper.get_all_layers(target_layer)
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        net_output = lasagne.layers.get_output(target_layer, deterministic=True)
        return theano.function(inputs=input_vars, outputs=net_output)


class SegmentationNetwork(Network):
    """
    Segmentation Neural Network
    """

    def predict_proba(self, input, squeeze=True, overlap=0.5):
        """
        Predict on test samples
        """
        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()

        # get network input shape
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        in_shape = l_in.output_shape[-2::]

        # standard prediction
        if input.shape[-2::] == in_shape:
            proba = self.compute_output(input)

        # sliding window prediction if images do not match
        else:
            proba = self._predict_proba_sliding_window(input, overlap=overlap)

        if squeeze:
            proba = proba.squeeze()

        return proba

    def predict(self, input, thresh=0.5):
        """
        Predict label map on test samples
        """
        P = self.predict_proba(input, squeeze=False)

        # binary segmentation
        if P.shape[1] == 1:
            return (P > thresh).squeeze()

        # categorical segmentation
        else:
            return np.argmax(P, axis=1).squeeze()

    def _predict_proba_sliding_window(self, images, overlap=0.5):
        """
        Sliding window prediction for images larger than the input layer
        """
        images = images.copy()
        n_images = images.shape[0]
        h, w = images.shape[2:4]
        _, Nc, sh, sw = self.net.output_shape

        # pad images for sliding window prediction
        missing_h = int(sh * np.ceil(float(h) / sh) - h)
        missing_w = int(sw * np.ceil(float(w) / sw) - w)

        pad_top = missing_h // 2
        pad_bottom = missing_h - pad_top

        pad_left = missing_w // 2
        pad_right = missing_w - pad_left

        images = np.pad(images, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        step_h = int(sh * (1.0 - overlap))
        row_0 = np.arange(0, images.shape[2] - sh + 1, step_h)
        row_1 = row_0 + sh

        step_w = int(sw * (1.0 - overlap))
        col_0 = np.arange(0, images.shape[3] - sw + 1, step_w)
        col_1 = col_0 + sw

        # import pdb
        # pdb.set_trace()

        # hamming window weighting
        window_h = np.hamming(sh)
        window_w = np.hamming(sw)
        ham2d = np.sqrt(np.outer(window_h, window_w))[np.newaxis, np.newaxis]

        # initialize result image
        R = np.zeros((n_images, Nc, images.shape[2], images.shape[3]))
        V = np.zeros((n_images, Nc, images.shape[2], images.shape[3]))

        for ir in range(len(row_0)):
            for ic in range(len(col_0)):
                I = images[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]]
                P = self.compute_output(I)
                R[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]] += P * ham2d
                V[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]] += ham2d

        # clip to original image size again
        R = R[:, :, pad_top:images.shape[2] - pad_bottom, pad_left:images.shape[3] - pad_right]
        V = V[:, :, pad_top:images.shape[2] - pad_bottom, pad_left:images.shape[3] - pad_right]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(V[0, 0])
        # plt.colorbar()
        # plt.show(block=True)

        # normalize predictions
        R /= V
        return R


class OpticalMusicRecognizer(object):
    """ Score Segmentation Networks """

    def __init__(self, note_detector=None, system_detector=None, bar_detector=None):
        self.note_detector = note_detector
        self.system_detector = system_detector
        self.bar_detector = bar_detector

        self.primitive_channel_mapping = dict()
        self.primitive_detector = dict()
        self.primitive_detector_ch = dict()

    def add_primitives_detector(self, primitives, detector=None, detector_ch=None):
        """ add primitive detector """

        if not isinstance(primitives, list):
            primitives = [primitives]

        for channel, primitive in enumerate(primitives):
            self.primitive_detector[primitive] = detector
            self.primitive_detector_ch[primitive] = detector_ch
            self.primitive_channel_mapping[primitive] = channel

    def detect_bars(self, image, systems=None, verbose=False):
        """ detect bars """
        import math
        from skimage.filters import threshold_otsu
        from skimage.morphology import label
        from skimage.measure import regionprops

        MIN_LENGTH = 80
        ANGLE_TOL = 5
        MIN_ECC = 0.95

        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis]
        bar_probs = self.bar_detector.predict_proba(image)

        if verbose:
            plt.figure("Bar Probabilities")
            plt.imshow(bar_probs, cmap=cmaps['magma'], vmin=0, vmax=1)

        # detect foreground
        fg_img = bar_probs > threshold_otsu(bar_probs)

        # compute region props
        label_img = label(fg_img, neighbors=8)
        region_props = regionprops(label_img)

        # find bars
        detected_bars = np.zeros((0, 2, 2))
        for blob in region_props:

            if blob.major_axis_length < MIN_LENGTH:
                continue

            if np.abs(90 - np.abs(math.degrees(blob.orientation))) > ANGLE_TOL:
                continue

            if blob.eccentricity < MIN_ECC:
                continue

            # shrink bounding box
            min_row, min_col, max_row, max_col = blob.bbox
            col = np.mean([min_col, max_col])

            # compile system coordinates
            bar_coords = np.zeros((2, 2))
            bar_coords[0] = np.asarray([min_row, col])
            bar_coords[1] = np.asarray([max_row, col])

            detected_bars = np.concatenate((detected_bars, bar_coords[np.newaxis]))

        # align bars with system
        if systems is not None:
            bars_by_system = self._bars_by_systems(detected_bars, systems)
            detected_bars = np.zeros((0, 2, 2))
            for i_sys, bars in enumerate(bars_by_system):

                # add missing bars
                if bars[0, 0, 1] != systems[i_sys, 0, 1]:
                    missing_bar = np.asarray([[systems[i_sys, 0, 0], systems[i_sys, 0, 1]],
                                              [systems[i_sys, 3, 1], systems[i_sys, 3, 1]]])
                    missing_bar = missing_bar[np.newaxis]
                    bars = np.vstack((missing_bar, bars))

                # add missing bars
                if np.abs(bars[0, 0, 1] - systems[i_sys, 0, 1]) > 10:
                    missing_bar = np.asarray([[systems[i_sys, 0, 0], systems[i_sys, 0, 1]],
                                              [systems[i_sys, 3, 1], systems[i_sys, 3, 1]]])
                    missing_bar = missing_bar[np.newaxis]
                    bars = np.vstack((missing_bar, bars))

                if np.abs(bars[-1, 0, 1] - systems[i_sys, 1, 1]) > 10:
                    missing_bar = np.asarray([[systems[i_sys, 1, 0], systems[i_sys, 1, 1]],
                                              [systems[i_sys, 2, 1], systems[i_sys, 2, 1]]])
                    missing_bar = missing_bar[np.newaxis]
                    bars = np.vstack((missing_bar, bars))

                # align bar with system
                for bar in bars:
                    bar[0, 0] = systems[i_sys, 0, 0]
                    bar[1, 0] = systems[i_sys, 3, 0]
                    detected_bars = np.concatenate((detected_bars, bar[np.newaxis]))

        return detected_bars

    def detect_notes(self, image, threshold_abs=0.5, min_distance=3, verbose=False):
        """ detect note heads """

        # predict note head probabilities
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis]
        note_probs = self.note_detector.predict_proba(image)

        if verbose:
            plt.figure("Note Probabilities")
            plt.imshow(note_probs, cmap=cmaps['magma'], vmin=0, vmax=1)
            plt.title("Note Probability Map", fontsize=22)

        # find local maxima (note heads)
        note_coords = peak_local_max(note_probs, min_distance=min_distance, threshold_abs=threshold_abs)
        return note_coords

    def detect_systems(self, image, verbose=False):
        """ detect systems """
        from skimage.filters import threshold_otsu
        from skimage.morphology import label
        from skimage.measure import regionprops

        MIN_AREA = 50000

        # predict note head probabilities
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis]
        system_probs = self.system_detector.predict_proba(image)

        # clean up space between systems
        if self.bar_detector:
            bar_probs = self.bar_detector.predict_proba(image)
            projection = bar_probs.sum(1)
        else:
            projection = system_probs.sum(1)

        thresh = threshold_otsu(projection)
        space_indices = np.nonzero(projection < thresh)[0]
        start_idx = prev_idx = space_indices[0]
        for idx in space_indices[1:]:
            if (idx - prev_idx) == 1:
                prev_idx = idx
            else:
                if prev_idx - start_idx > 15:
                    system_probs[start_idx:prev_idx, :] = 0
                start_idx = prev_idx = idx

        # detect foreground
        fg_img = system_probs > threshold_otsu(system_probs)

        # morphologically close foreground image
        kernel = np.asarray([1] * 15).reshape((15, 1))
        fg_img = cv2.morphologyEx(fg_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        if verbose:
            plt.figure("Sheet Image")
            plt.imshow(image[0, 0], cmap="gray", vmin=0, vmax=1)

            plt.figure("System Probabilities")
            plt.imshow(system_probs, cmap=cmaps['magma'], vmin=0, vmax=1)

            plt.figure("FG Image")
            plt.imshow(fg_img, cmap=plt.cm.gray, vmin=0, vmax=1)

            plt.figure("Projection")
            plt.plot(projection)

        # compute region props
        label_img = label(fg_img, neighbors=8)
        region_props = regionprops(label_img)

        # find systems
        detected_systems = np.zeros((0, 4, 2))
        for blob in region_props:

            if blob.area < MIN_AREA:
                continue

            # shrink bounding box
            bbox = self._shrink_bounding_box(label_img == blob.label, blob.bbox)
            min_row, min_col, max_row, max_col = bbox

            # snap system to grid
            min_row, max_row, min_col, max_col = snap_system_to_grid(image, min_row, max_row, min_col, max_col)

            # compile system coordinates
            system_coords = np.zeros((4, 2))
            system_coords[0] = np.asarray([min_row, min_col])
            system_coords[1] = np.asarray([min_row, max_col])
            system_coords[2] = np.asarray([max_row, max_col])
            system_coords[3] = np.asarray([max_row, min_col])

            detected_systems = np.concatenate((detected_systems, system_coords[np.newaxis]))

        return detected_systems

    def detect_systems_ly(self, image, verbose=False):

        # pre-process of images
        image = (image <= 0.5).astype(np.uint8)
        kernel_size = int(image.shape[1] * 0.7)
        kernel = np.ones((1, kernel_size), dtype=np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        plt.figure()
        plt.imshow(image)
        plt.show()

        label_img = label(image)
        blobs = regionprops(label_img)

        labels = np.unique(label_img)
        num_labels = len(labels)
        print(float(num_labels - 1) / 10)

        for g, blob in enumerate(blobs):
            target_label = ((blob.label - 1) // 10) + 1
            label_img[label_img == blob.label] = target_label

        system_blobs = regionprops(label_img)
        detected_systems = np.zeros((0, 4, 2))
        for blob in system_blobs:
            min_row, min_col, max_row, max_col = blob.bbox

            # compile system coordinates
            system_coords = np.zeros((4, 2))
            system_coords[0] = np.asarray([min_row, min_col])
            system_coords[1] = np.asarray([min_row, max_col])
            system_coords[2] = np.asarray([max_row, max_col])
            system_coords[3] = np.asarray([max_row, min_col])

            detected_systems = np.concatenate((detected_systems, system_coords[np.newaxis]))

        return detected_systems

    def detect_primitives(self, image, primitive, threshold_abs=0.5, kernel_size=3, detector='mask', verbose=False,
                          return_labels=False):
        """
        general omr primitive detector

        detector: mask, conv_hull, combined
        """
        from skimage.measure import label, regionprops

        # prepare images
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis]

        # compute probability maps and segmentation
        if detector in ['mask', 'combined']:
            prob_map = self.primitive_detector[primitive].predict_proba(image, squeeze=False)[0]

            # select output channel
            channel = self.primitive_channel_mapping[primitive]
            prob_map = prob_map[channel]

            binary = prob_map > threshold_abs

        if detector in ['conv_hull', 'combined']:
            prob_map_ch = self.primitive_detector_ch[primitive].predict_proba(image, squeeze=False)[0]

            # select output channel
            channel = self.primitive_channel_mapping[primitive]
            prob_map_ch = prob_map_ch[channel]

            binary_ch = (prob_map_ch > threshold_abs).astype(np.uint8)

            # close holes in convex hull label
            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                binary_ch = cv2.morphologyEx(binary_ch, cv2.MORPH_CLOSE, kernel)

        # compute label image
        if detector in ['conv_hull', 'combined']:
            label_img_ch = label(binary_ch)

            if detector == "combinied":
                # propagate labels to primitive image
                label_img = binary.copy().astype(np.int64)
                label_img *= label_img_ch
            else:
                label_img = label_img_ch

        else:
            label_img = label(binary)

        # collect centroids
        regions = regionprops(label_img)
        centroids = []
        for r in regions:
            if r.area <= 1:
                label_img[label_img == r.label] = 0
                continue
            centroids.append(r.centroid)
        centroids = np.asarray(centroids)

        if verbose:
            plt.figure("Primitive Detector")
            plt.clf()

            ax1 = plt.subplot(3, 2, 6)
            plt.imshow(image[0, 0], cmap=plt.cm.gray)
            if len(centroids) > 0:
                plt.plot(centroids[:, 1], centroids[:, 0], 'mo', alpha=0.5)
            plt.title("Detection Result")

            if detector in ['mask', 'combined']:
                plt.subplot(3, 2, 1, sharex=ax1, sharey=ax1)
                plt.imshow(prob_map, cmap=cmaps['magma'], vmin=0, vmax=1)
                plt.title("Probability Map")

                plt.subplot(3, 2, 3, sharex=ax1, sharey=ax1)
                plt.imshow(binary, cmap=plt.cm.gray, vmin=0, vmax=1)
                plt.title("Primitive Binary Image")

                plt.subplot(3, 2, 5, sharex=ax1, sharey=ax1)
                plt.imshow(label_img, cmap='jet')
                plt.title("Primitive Labels")

            if detector in ['conv_hull', 'combined']:
                plt.subplot(3, 2, 2, sharex=ax1, sharey=ax1)
                plt.imshow(prob_map_ch, cmap=cmaps['magma'], vmin=0, vmax=1)
                plt.title("Convex Hull Probability Map")

                plt.subplot(3, 2, 4, sharex=ax1, sharey=ax1)
                plt.imshow(label_img_ch, cmap='jet')
                plt.title("Convex Hull Label Image")

            # plt.show(block=True)

        if return_labels:
            return centroids, label_img
        else:
            return centroids

    def _shrink_bounding_box(self, fg_img, bbox):
        """ shrink bounding box """

        min_row, min_col, max_row, max_col = bbox

        # clip regions to image size
        min_row = max(min_row, 0)
        min_col = max(min_col, 0)
        max_row = min(max_row, fg_img.shape[0] - 1)
        max_col = min(max_col, fg_img.shape[1] - 1)

        while np.mean(fg_img[min_row, min_col:max_col]) < 0.9:
            min_row += 1

        while np.mean(fg_img[max_row, min_col:max_col]) < 0.9:
            max_row -= 1

        while np.mean(fg_img[min_row:max_row, min_col]) < 0.9:
            min_col += 1

        while np.mean(fg_img[min_row:max_row, max_col]) < 0.9:
            max_col -= 1

        return min_row, min_col, max_row, max_col

    def _bars_by_systems(self, page_bars, page_systems):
        """ assign bars to systems """
        from sklearn.metrics.pairwise import pairwise_distances

        # compute y-coordinates
        page_systems_centers = page_systems.mean(1)[:, 0:1]
        page_bar_centers = page_bars.mean(1)[:, 0:1]

        # compute pairwise distances
        dists = pairwise_distances(page_bar_centers, page_systems_centers)

        # assign bars to systems
        bars_by_system = [np.zeros((0, 2, 2))] * page_systems.shape[0]
        for i in range(dists.shape[0]):
            min_idx = np.argmin(dists[i])
            bars = page_bars[i][np.newaxis, :, :]
            bars_by_system[min_idx] = np.vstack((bars_by_system[min_idx], bars))

        # sort bars from left to right
        for i in range(page_systems.shape[0]):
            sorted_idx = np.argsort(bars_by_system[i][:, 0, 1])
            bars_by_system[i] = bars_by_system[i][sorted_idx]

        return bars_by_system
