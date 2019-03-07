#!/usr/bin/env python
# author: matthias dorfer

from __future__ import print_function

import os
import importlib
import pickle
import lasagne
import argparse

from config.settings import EXP_ROOT
from utils import mutopia_data

# init color printer
from utils.plotting import BColors
col = BColors()


def select_model(model_path):
    """ select model and train function """

    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]

    model = importlib.import_module('models.' + model_str)

    from utils.train_dcca_pool import fit

    model.EXP_NAME = model_str
    return model, fit


def select_data(data_name, split_file, config_file, seed=23, test_only=False):
    """ select train data """

    if str(data_name) == "mutopia":
        data = mutopia_data.load_audio_score_retrieval(split_file=split_file, config_file=config_file,
                                                       test_only=test_only)
    else:
        pass

    return data


def compile_tag(train_split, config):
    """ compile model tag fom split and config file paths """
    tag = os.path.splitext(os.path.basename(train_split))[0]
    tag += "_" + os.path.splitext(os.path.basename(config))[0]
    return tag


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train cross-modality retrieval model.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select data for training.')
    parser.add_argument('--resume', help='resume on pre-trained model.', action='store_true')
    parser.add_argument('--seed', help='query direction.', type=int, default=23)
    parser.add_argument('--no_dump', help='do not dump model file.', action='store_true')
    parser.add_argument('--show_architecture', help='print model architecture.', action='store_true')
    parser.add_argument('--train_split', help='path to train split file.', type=str, default=None)
    parser.add_argument('--config', help='path to experiment config file.', type=str, default=None)
    args = parser.parse_args()

    # select model
    model, fit = select_model(args.model)

    # check if cca should be trained on network output
    fit_cca = model.FIT_CCA if hasattr(model, 'FIT_CCA') else True

    # check if refinement patience is set
    refinement_patience = getattr(model, 'REFINEMENT_PATIENCE', 10)

    # check if model should be pretrained
    pretrain_epochs = getattr(model, 'PRETRAIN_EPOCHS', 0)

    # select data
    print("\nLoading data...")
    data = select_data(args.data, args.train_split, args.config, args.seed)

    # tag parameter file
    tag = compile_tag(args.train_split, args.config)
    print("Experimental Tag:", tag)

    # set model dump file
    config_str = os.path.splitext(os.path.split(args.config)[1])[0]
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME + '_' + config_str)
    dump_file = 'params.pkl' if tag is None else 'params_%s.pkl' % tag
    dump_file = os.path.join(out_path, dump_file)
    log_file = 'results.pkl' if tag is None else 'results_%s.pkl' % tag
    log_file = os.path.join(out_path, log_file)

    print("\nBuilding network...")
    layers = model.build_model(input_shape_1=[1, data['train'].staff_height, data['train'].sheet_context],
                               input_shape_2=[1, data['train'].spec_bins, data['train'].spec_context],
                               show_model=args.show_architecture)

    if args.resume:
        print("\n")
        print("Loading model parameters from:", dump_file)
        with open(dump_file, 'r') as fp:
            params = pickle.load(fp)
            lasagne.layers.set_all_param_values(layers, params)

    # do not dump model
    dump_file = None if args.no_dump else dump_file

    # train model
    # -----------
    train_batch_iter = model.train_batch_iterator(model.BATCH_SIZE)
    valid_batch_iter = model.valid_batch_iterator()
    layers, va_loss = fit(layers, data, model.objectives,
                          train_batch_iter=train_batch_iter, valid_batch_iter=valid_batch_iter,
                          num_epochs=model.MAX_EPOCHS, patience=model.PATIENCE,
                          learn_rate=model.INI_LEARNING_RATE, update_learning_rate=model.update_learning_rate,
                          compute_updates=model.compute_updates, l_2=model.L2, l_1=model.L1,
                          exp_name=model.EXP_NAME, out_path=out_path, dump_file=dump_file, log_file=log_file,
                          fit_cca=fit_cca, pretrain_epochs=pretrain_epochs,
                          refinement_steps=model.REFINEMENT_STEPS, lr_multiplier=model.LR_MULTIPLIER,
                          refinement_patience=refinement_patience)
