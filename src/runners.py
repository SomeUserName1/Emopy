import argparse
import os

import data_collect.ck_structure as ck_collect
import data_collect.fer2013_structure as fer_collect
from config import *
from keras_models.dlib_inputs import DlibPointsInputNeuralNet
from keras_models.img_input import NeuralNet
from keras_models.multinput import MultiInputNeuralNet
from keras_models.rnn import LSTMNet, DlibLSTMNet
from preprocess.base import Preprocessor
from preprocess.dlib_input import DlibInputPreprocessor
from preprocess.multinput import MultiInputPreprocessor
from preprocess.sequencial import SequencialPreprocessor, DlibSequencialPreprocessor
from util.ClassifierWrapper import SevenEmotionsClassifier

maxSequenceLength = 10


def run(shape_predictor_path, data_set_dir, data_out_dir, model_out_dir, net_type,
        session, img_size, lr, steps, batch_size, epochs, augmentation,
        pred_img, pred_vid, pred_type):
    """

    """
    input_shape = (img_size[0], img_size[1], 1)
    classifier = SevenEmotionsClassifier()

    if NETWORK_TYPE == "imnn":
        preprocessor = Preprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
        neural_net = NeuralNet(input_shape, preprocessor=preprocessor, train=True)

    elif NETWORK_TYPE == "dinn":
        preprocessor = DlibInputPreprocessor(classifier, shape_predictor_path, input_shape=input_shape,
                                             augmentation=augmentation)
        neural_net = DlibPointsInputNeuralNet(input_shape, preprocessor=preprocessor, train=True)

    elif NETWORK_TYPE == "minn":
        preprocessor = MultiInputPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
        neural_net = MultiInputNeuralNet(input_shape, preprocessor=preprocessor, train=True)

    elif NETWORK_TYPE == "lstm":
        preprocessor = SequencialPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)(
            "dataset/ck-split")
        neural_net = LSTMNet(input_shape, preprocessor=preprocessor, train=True)

    elif NETWORK_TYPE == "dlstm":
        preprocessor = DlibSequencialPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)(
            "dataset/ck-split")
        neural_net = DlibLSTMNet(input_shape, preprocessor=preprocessor, train=True)

    # elif NETWORK_TYPE == "milstm":
    #    preprocessor = MultiInputPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
    #   neural_net = MultiInputLSTMNet

    # elif NETWORK_TYPE == "caps":
    #    preprocessor = MultiInputPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
    #   neural_net = MultiInputLSTMNet

    else:
        raise Exception("Network type must be in {mi for a multi-input NN, si for a single-input NN, rnn for LSTM "
                        ", drnn for sequential using the CK data set, dinn for a single Dlib input net }")

    print("runners.run()")
    if session == 'init_data':
        ck_collect.main(data_set_dir, data_out_dir)
        fer_collect.main(data_set_dir, data_out_dir)
    if session == 'train':
        neural_net.train()
    if session == 'test':
        neural_net.test()
    elif SESSION == 'predict':
        neural_net.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Directory Setup
    parser.add_argument("--shape_predictor_path", default=SHAPE_PREDICTOR_PATH, type=str)
    parser.add_argument("--data_set_dir", default=DATA_SET_DIR, type=str)
    parser.add_argument("--data_out_dir", default=DATA_OUT_DIR, type=str)
    parser.add_argument("--model_out_dir", default=MODEL_OUT_PATH, type=str)

    # Session Setup
    parser.add_argument("--net_type", default=NETWORK_TYPE, type=str)
    parser.add_argument("--session", default=SESSION, type=str)
    parser.add_argument("--img_size", default=IMG_SIZE, type=(int, int))

    # Training and Testing setup
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--steps", default=STEPS_PER_EPOCH, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--augmentation", default=AUGMENTATION, type=bool)

    # Prediction Setup
    parser.add_argument("--pred_img", default=MODEL_OUT_PATH, type=str)
    parser.add_argument("--pred_vid", default=MODEL_OUT_PATH, type=str)
    parser.add_argument("--pred_type", default=MODEL_OUT_PATH, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_set_dir):
        print("Data set path given does not exists")
        exit(0)

    run(args.shape_predictor_path, args.data_set_dir, args.data_out_dir, args.model_out_dir, args.net_type,
        args.session, args.img_size, args.lr, args.steps, args.batch_size, args.epochs, args.augmentation,
        args.pred_img, args.pred_vid, args.pred_type)
