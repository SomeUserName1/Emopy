import argparse
import os

from keras_models.rnn.lstm import LSTMNet, DlibLSTMNet

import data_collect.ck_structure as ck_collect
import data_collect.fer2013_structure as fer_collect
from config import *
from keras_models.caps.caps import CapsNet
from keras_models.cnn.dlib_inputs import DlibPointsInputNeuralNet
from keras_models.cnn.img_input import ImageInputNeuralNet
from keras_models.cnn.multinput import MultiInputNeuralNet
from preprocess.dlib_input import DlibInputPreprocessor
from preprocess.image_input import Preprocessor
from preprocess.multinput import MultiInputPreprocessor
from preprocess.sequencial import SequencialPreprocessor, DlibSequencialPreprocessor
from util.ClassifierWrapper import SevenEmotionsClassifier

maxSequenceLength = 10


def run(shape_predictor_path, data_set_dir, data_out_dir, model_out_dir, net_type,
        session, img_size, logger, lr, batch_size, steps, epochs, augmentation,
        pred_img, pred_vid, pred_type):
    """

    """
    if session == "'init_data'":
        ck_collect.main(data_set_dir, data_out_dir)
        fer_collect.main(data_set_dir, data_out_dir)
        print("finished generating data")
        return

    input_shape = (img_size[0], img_size[1], 1)
    classifier = SevenEmotionsClassifier()

    if net_type == "imagenn":
        preprocessor = Preprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
        neural_net = ImageInputNeuralNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps,
                                         epochs,
                                         preprocessor, logger, session)

    elif net_type == "dlibnn":
        preprocessor = DlibInputPreprocessor(classifier, shape_predictor_path, input_shape=input_shape,
                                             augmentation=augmentation)
        neural_net = DlibPointsInputNeuralNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps,
                                              epochs,
                                              preprocessor, logger, session)

    elif net_type == "minn":
        preprocessor = MultiInputPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
        neural_net = MultiInputNeuralNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps,
                                         epochs,
                                         preprocessor, logger, session)

    elif net_type == "imlstm":
        preprocessor = SequencialPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)(
            "dataset/ck-split")
        neural_net = LSTMNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                             preprocessor, logger, session)

    elif net_type == "dliblstm":
        preprocessor = DlibSequencialPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation,
                                                  predictor_path=shape_predictor_path)(
            "dataset/ck-split")
        neural_net = DlibLSTMNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                                 preprocessor, logger, session)

    # elif net_type == "milstm":
    #    preprocessor = MultiInputPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
    #   neural_net = MultiInputLSTMNet

    elif net_type == "caps":
        preprocessor = Preprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
        neural_net = CapsNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                             preprocessor, logger, session, lmd=0.5)

    else:
        raise Exception("Network type must be in {mi for a multi-input NN, si for a single-input NN, rnn for LSTM "
                        ", drnn for sequential using the CK data set, dinn for a single Dlib input net }")

    print("runners.run()")

    if session == "'train'":
        neural_net.train()
    if session == "'predict'":
        neural_net.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Directory Setup
    parser.add_argument("--shape_predictor_path", default=SHAPE_PREDICTOR_PATH, type=str)
    parser.add_argument("--data_set_dir", default=DATA_SET_DIR, type=str)
    parser.add_argument("--data_out_dir", default=DATA_OUT_DIR, type=str)
    parser.add_argument("--model_out_dir", default=MODEL_OUT_DIR, type=str)

    # Session Setup
    parser.add_argument("--net_type", default=NETWORK_TYPE, type=str)
    parser.add_argument("--session", default=SESSION, type=str)
    parser.add_argument("--img_size", default=IMG_SIZE, type=int, nargs=2)
    parser.add_argument("--logger", default=None, type=str)

    # Training and Testing setup
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--steps", default=STEPS_PER_EPOCH, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--augmentation", default=AUGMENTATION, type=bool)

    # Prediction Setup
    parser.add_argument("--pred_img", default=PREDICTION_IMAGE, type=str)
    parser.add_argument("--pred_vid", default=PREDICTION_VIDEO, type=str)
    parser.add_argument("--pred_type", default=PREDICTION_TYPE, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_set_dir):
        raise Exception("Data set path given does not exists")

    run(args.shape_predictor_path, args.data_set_dir, args.data_out_dir, args.model_out_dir, args.net_type,
        args.session, args.img_size, args.logger, args.lr, args.batch_size, args.steps, args.epochs, args.augmentation,
        args.pred_img, args.pred_vid, args.pred_type)
