import argparse
import os

import cv2
import dlib
import numpy as np

import data_collect.ck_structure as ck_collect
import data_collect.fer2013_structure as fer_collect
from config import *
from keras_models.caps.caps import CapsNet
from keras_models.cnn.dlib_inputs import DlibPointsInputNeuralNet
from keras_models.cnn.img_input import ImageInputNeuralNet
from keras_models.cnn.inception_resnet import InceptionResNet
from keras_models.cnn.multinput import MultiInputNeuralNet
from preprocess.CapsPreprocessor import CapsPreprocessor
from preprocess.DlibPreprocessor import DlibInputPreprocessor
from preprocess.ImagePreprocessor import Preprocessor
from preprocess.MultiPreprocessor import MultiInputPreprocessor
from util.ClassifierWrapper import SevenEmotionsClassifier
from util.PostProcessor import PostProcessor

# from keras_models.lstm.dliblstm import DlibLSTMNet
# from keras_models.lstm.lstm import LSTMNet
# from preprocess.DlibSequencePreprocessor import DlibSequencialPreprocessor
# from preprocess.SequencePreprocessor import SequencialPreprocessor

maxSequenceLength = 10


def run(shape_predictor_path, data_set_dir, data_out_dir, model_out_dir, net_type,
        session, img_size, logger, lr, batch_size, steps, epochs, augmentation,
        pred_data, pred_type):
    """

    """
    if session == "'init_data'":
        ck_collect.main(data_set_dir, data_out_dir)
        fer_collect.main(data_set_dir, data_out_dir)
        print("finished generating data")
        return

    input_shape = (img_size[0], img_size[1], 1)
    classifier = SevenEmotionsClassifier()

    if session == "'predict'":
        data_out_dir = pred_data
        batch_size = 1
        steps = 1
        epochs = 1

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
        preprocessor = MultiInputPreprocessor(classifier, shape_predictor_path, input_shape=input_shape,
                                              augmentation=augmentation)
        neural_net = MultiInputNeuralNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps,
                                         epochs,
                                         preprocessor, logger, session)

    elif net_type == "caps":
        preprocessor = CapsPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
        neural_net = CapsNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                             preprocessor, logger, session, lmd=0.5)

    elif net_type == "incresnet":
        preprocessor = MultiInputPreprocessor(classifier, shape_predictor_path, input_shape=input_shape,
                                              augmentation=augmentation)
        neural_net = InceptionResNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                                     preprocessor, logger, session)

    # elif net_type == "imlstm":
    #     preprocessor = SequencialPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)(
    #         "dataset/ck-split")
    #     neural_net = LSTMNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
    #                          preprocessor, logger, session)
    #
    # elif net_type == "dliblstm":
    #     preprocessor = DlibSequencialPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation,
    #                                               predictor_path=shape_predictor_path)(
    #         "dataset/ck-split")
    #     neural_net = DlibLSTMNet(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
    #                              preprocessor, logger, session)

    # elif net_type == "milstm":
    #    preprocessor = MultiInputPreprocessor(classifier, input_shape=input_shape, augmentation=augmentation)
    #   neural_net = MultiInputLSTMNet

    else:
        raise Exception("Network type must be in {mi for a multi-input NN, si for a single-input NN, rnn for LSTM "
                        ", drnn for sequential using the CK data set, dinn for a single Dlib input net }")

    print("runners.run()")

    if session == "'train'":
        neural_net.train()

    if session == "'predict'":
        face_detector = dlib.get_frontal_face_detector()
        post_processor = PostProcessor(classifier)

        if pred_type == "image":
            img = cv2.imread(pred_data)
            faces, rectangles = preprocessor.get_faces(img, face_detector)
            predictions = []
            for i in range(len(faces)):
                face = preprocessor.sanitize(faces[i])
                predictions.append(neural_net.predict(face))

            post_processor(img, rectangles, predictions)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif pred_type == "video":
            cap = cv2.VideoCapture(pred_data)
            while cap.isOpened():
                ret, frame = cap.read()
                current_width = frame.shape[1]
                width = 600
                ratio = current_width / float(width)
                height = frame.shape[0] / float(ratio)
                frame = cv2.resize(frame, (width, int(height)))
                faces, rectangles = preprocessor.get_faces(frame, face_detector)
                if len(faces) > 0:
                    emotions = []
                    for i in range(len(faces)):
                        face = preprocessor.sanitize(faces[i]).astype(np.float32) / 255
                        predictions = neural_net.predict(face.reshape(-1, 64, 64, 1))[0]
                        print(predictions)
                        emotions.append(classifier.get_string(arg_max(predictions)))

                    post_processor.overlay(frame, rectangles, emotions)
                cv2.imshow("Video", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        elif pred_type == "web_cam":
            rectangles = None
            cap = cv2.VideoCapture(-1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            sequences = np.zeros(
                (maxSequenceLength, input_shape[0], input_shape[1], input_shape[2]))
            while cap.isOpened():
                while len(sequences) < maxSequenceLength:
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (300, 240))
                    faces, rectangles = preprocessor.get_faces(frame, face_detector)
                predictions = []
                for i in range(len(faces)):
                    face = preprocessor.sanitize(faces[i])
                    predictions.append(neural_net.predict(face))

                post_processor.overlay(frame, rectangles, predictions)
                cv2.imshow("Image", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                cv2.destroyAllWindows()


def show_sequence_images(path):
    """

    Args:
        path:
    """
    for img_file in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_file))
        cv2.imshow("Sequence", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def arg_max(array):
    """

    Args:
        array:

    Returns:

    """
    max_value = array[0]
    index = 0
    for i, el in enumerate(array):
        if el > max_value:
            index = i
            max_value = el
    return index


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
    parser.add_argument("--pred_data", default=PREDICTION_DATA, type=str)
    parser.add_argument("--pred_type", default=PREDICTION_TYPE, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_set_dir):
        raise Exception("Data set path given does not exists")

    run(args.shape_predictor_path, args.data_set_dir, args.data_out_dir, args.model_out_dir, args.net_type,
        args.session, args.img_size, args.logger, args.lr, args.batch_size, args.steps, args.epochs, args.augmentation,
        args.pred_data, args.pred_type)
