import os

import cv2
import dlib
import numpy as np
from keras.models import model_from_json
from keras_models.rnn.rnn import DlibLSTMNet

from config import *
from keras_models.cnn.img_input import NeuralNet
from keras_models.cnn.multinput import MultiInputNeuralNet
from preprocess.base import Preprocessor
from preprocess.sequencial import DlibSequencialPreprocessor
from util.BasePostprocessor import PostProcessor
from util.ClassifierWrapper import SevenEmotionsClassifier

# from multiprocessing.queues import Queue
# from threading import Thread

maxSequenceLength = 10

def load_model(path):
    """

    Args:
        path:

    Returns:

    """
    with open(path + ".json") as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(path + ".h5")
        return model


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


def test_sequences():
    """

    """
    # TODO ck-split info gathering
    # dataset_path = "dataset/ck-split/test"
    # for emotion_dir in os.listdir(dataset_path):
    raise NotImplementedError


# TODO Move to appropriate NN class
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


def draw_landmarks(frame, landmarks):
    """

    Args:
        frame:
        landmarks:
    """
    for i in range(len(landmarks)):
        landmark = landmarks[i]
        cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, color=(255, 0, 0), thickness=1)


def predict():
    """

    """
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
    classifier = SevenEmotionsClassifier()
    preprocessor = Preprocessor(classifier, input_shape=input_shape)
    postProcessor = PostProcessor(classifier)
    neural_net = MultiInputNeuralNet(input_shape, preprocessor=preprocessor, learning_rate=1e-4, batch_size=1,
                                     epochs=100, steps_per_epoch=1,
                                     dataset_dir=PREDICTION_IMAGE, train=False)
    face_detector = dlib.get_frontal_face_detector()

    if PREDICTION_TYPE == "image":
        img = cv2.imread(PREDICTION_IMAGE)
        faces, rectangles = preprocessor.get_faces(img, face_detector)
        predictions = []
        for i in range(len(faces)):
            face = preprocessor.sanitize(faces[i])
            predictions.append(neural_net.predict(face))
        print("predicted")

        postProcessor = postProcessor(img, rectangles, predictions)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif PREDICTION_TYPE == "video":  # TODO
        pass
    elif PREDICTION_TYPE == "webcam":
        if NETWORK_TYPE == "drnn":
            # cap = cv2.VideoCapture("/home/mtk/iCog/projects/emopy/test-videos/75Emotions.mp4")
            cap = cv2.VideoCapture(-1)
            preprocessor = DlibSequencialPreprocessor(classifier, input_shape=input_shape)
            neural_net = DlibLSTMNet(input_shape, preprocessor=preprocessor, train=False)
            face_detector = dlib.get_frontal_face_detector()
            postProcessor = PostProcessor(classifier)
            # TODO Why is multi-threadding uncommented
            # if cap.isOpened():
            #     recognitionThread = Thread(target=start_recognition_task,args=(preprocessor,neural_net))
            #     recognitionThread.start()
            print
            "opening camera"

            current_sequence = np.zeros((maxSequenceLength, 68, 2, 1))
            currentIndex = 0

            currentEmotion = ""
            while cap.isOpened():
                ret, frame = cap.read()
                currentWidth = frame.shape[1]
                width = 600
                ratio = currentWidth / float(width)
                height = frame.shape[0] / float(ratio)
                frame = cv2.resize(frame, (width, int(height)))
                faces, rectangles = preprocessor.get_faces(frame, face_detector)
                if (len(faces) > 0):
                    face, rectangle = faces[0], rectangles[0]
                    face = preprocessor.sanitize(face)
                    dlib_points = preprocessor.get_face_dlib_points(face)
                    # draw_landmarks(face,dlib_points)
                    # cv2.imshow("Face",face)
                    current_sequence[currentIndex:currentIndex + 2] = [np.array(np.expand_dims(dlib_points, 2)),
                                                                       np.expand_dims(dlib_points, 2)]
                    currentIndex += 2
                    # sequencialQueue.put(face)
                    postProcessor.overlay(frame, [rectangle], [currentEmotion])
                else:
                    current_sequence = np.zeros((maxSequenceLength, 68, 2, 1))
                    currentIndex = 0
                if currentIndex > maxSequenceLength - 2:
                    current_sequence = current_sequence.astype(np.float32) / IMG_SIZE[0]
                    predictions = neural_net.predict(np.expand_dims(current_sequence, 0))[0]
                    print
                    predictions
                    emotion = arg_max(predictions)
                    currentEmotion = preprocessor.classifier.get_string(emotion)
                    current_sequence = np.zeros((maxSequenceLength, 68, 2, 1))
                    currentIndex = 0
                cv2.imshow("Webcam", frame)
                if (cv2.waitKey(10) & 0xFF == ord('q')):
                    break
            cv2.destroyAllWindows()
        else:
            # TODO fix static path
            input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
            classifier = SevenEmotionsClassifier()
            preprocessor = Preprocessor(classifier, input_shape=input_shape)
            postProcessor = PostProcessor(classifier)
            neural_net = NeuralNet(input_shape, preprocessor=preprocessor, train=False)
            face_detector = dlib.get_frontal_face_detector()
            neural_net.load_model(MODEL_OUT_PATH + 'minn/minn-0')
            # cap = cv2.VideoCapture(-1)
            cap = cv2.VideoCapture("/home/mtk/iCog/projects/emopy/test-videos/75Emotions.mp4")
            while cap.isOpened():
                ret, frame = cap.read()
                currentWidth = frame.shape[1]
                width = 600
                ratio = currentWidth / float(width)
                height = frame.shape[0] / float(ratio)
                frame = cv2.resize(frame, (width, int(height)))
                faces, rectangles = preprocessor.get_faces(frame, face_detector)
                if (len(faces) > 0):
                    emotions = []
                    for i in range(len(faces)):
                        print
                        faces[i].shape
                        face = preprocessor.sanitize(faces[i]).astype(np.float32) / 255;
                        print
                        face.shape
                        predictions = neural_net.predict(face.reshape(-1, 48, 48, 1))[0]
                        print
                        predictions
                        emotions.append(classifier.get_string(arg_max(predictions)))

                    postProcessor.overlay(frame, rectangles, emotions)
                cv2.imshow("Webcam", frame)
                if (cv2.waitKey(10) & 0xFF == ord('q')):
                    break
            cv2.destroyAllWindows()



# ------------------ Lost and Found -----------------
#        self.train_image_paths = np.array(self.train_image_paths)
#        self.test_images = self.feature_extractor.extract(self.test_images)
#        self.test_image_emotions = np.eye(self.classifier.get_num_class())[np.array(self.test_image_emotions)]
#        self.test_image_paths = []
#        self.train_image_emotions = np.array(self.train_image_emotions)