import cv2
import keras
import numpy as np
from keras.layers import Flatten
from keras.layers import LSTM, Dense, Conv2D, TimeDistributed
from keras.models import Sequential

from keras_models.base import AbstractNet


def proc_webcam():
    cap = cv2.VideoCapture(-1)

    print("opening camera")

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
            print(predictions)
            emotion = arg_max(predictions)
            currentEmotion = preprocessor.classifier.get_string(emotion)
            current_sequence = np.zeros((maxSequenceLength, 68, 2, 1))
            currentIndex = 0
        cv2.imshow("Webcam", frame)
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()


class DlibLSTMNet(AbstractNet):
    """
    """

    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs, preprocessor, logger, session, post_processor=None):
        super(DlibLSTMNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size,
                                          steps_per_epoch, epochs, preprocessor, logger, session)
        self.TAG = "dliblstm"
        self.max_sequence_length = 10
        self.postProcessor = post_processor
        self.feature_extractors = ['dlib']
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        super(DlibLSTMNet, self).init_model(self.session)

    def build(self):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(64, (3, 1), padding='valid', activation='relu'),
                                  input_shape=(self.max_sequence_length, 68, 2, 1)))
        model.add(TimeDistributed(Conv2D(128, (3, 1), padding='valid', activation='relu')))
        # model.add(TimeDistributed(Dropout(0.5)))
        # model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(32, return_sequences=True, stateful=False))
        model.add(LSTM(64, return_sequences=False, stateful=False))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(6, activation="softmax"))

        return model

    def train(self):
        assert self.model is not None, "Model not built yet."
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

        self.preprocessor = self.preprocessor(self.data_dir)

        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=(
                                     self.preprocessor.test_sequences_dpoints, self.preprocessor.test_sequence_labels))

        score = self.model.evaluate(self.preprocessor.test_sequences_dpoints, self.preprocessor.test_sequence_labels)

        self.save_model()
        self.logger.log_model(self.TAG, score, self.model)

    def predict(self, dlib_features):
        emotions = self.model.predict(dlib_features)
        return emotions
