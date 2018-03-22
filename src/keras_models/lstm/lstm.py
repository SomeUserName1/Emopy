import cv2
import keras
from keras.layers import LSTM, Dense, Conv2D, TimeDistributed
from keras.layers import MaxPooling2D, Flatten
from keras.models import Sequential, model_from_json

from keras_models.base import AbstractNet


class LSTMNet(AbstractNet):
    """
    """

    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs, preprocessor, logger, session, post_processor=None):
        super(LSTMNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size,
                                      steps_per_epoch, epochs, preprocessor, logger, session)
        self.max_sequence_length = 10
        self.postProcessor = post_processor
        self.feature_extractors = ['image']
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        super(LSTMNet, self).init_model(self.session)

    def build(self):
        """

        Returns:

        """

        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='valid', activation='relu'),
                                  input_shape=(self.max_sequence_length, 48, 48, 1)))
        # model.add(TimeDistributed(Conv2D(64,(3,3),padding="valid",activation="relu")))
        # model.add(TimeDistributed(Dropout(0.2)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        # model.add(Bidirectional(LSTM(128,return_sequences=False,stateful=False,activation="relu",recurrent_dropout=0.2)))
        model.add(LSTM(64, return_sequences=False, stateful=False, activation="relu", recurrent_dropout=0.2))
        # model.add(Dropout(0.2))
        # model.add(Dense(128,activation="relu"))
        # model.add(Dropout(0.2))
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
                                     self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels))

        score = self.model.evaluate(self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels)

        self.save_model()
        self.logger.log_model(self.net_type, score, self.model)

    def predict(self, sequence_faces):
        """

        Args:
            sequence_faces:

        Returns:

        """
        assert sequence_faces[0].shape == self.input_shape, "Face image size should be " + str(self.input_shape)
        face = sequence_faces.reshape(-1, self.max_sequence_length, 48, 48, 1)
        emotions = self.model.predict(face)[0]
        return emotions

    def process_web_cam(self):
        model = model_from_json(open("models/rnn/rnn-0.json").read())
        model.load_weights("models/rnn/rnn-0.h5")
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        sequences = np.zeros((self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        while cap.isOpened():
            while len(sequences) < self.max_sequence_length:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (300, 240))
                faces, rectangles = self.preprocessor.get_faces(frame, face_detector)
                face = faces[0]
                sequences
            predictions = []
            for i in range(len(faces)):
                face = preprocessor.sanitize(faces[i])
                predictions.append(neuralNet.predict(face))

            self.postProcessor = self.postProcessor(img, rectangles, predictions)
            cv2.imshow("Image", img)
            if (cv2.waitKey(10) & 0xFF == ord('q')):
                break
            cv2.destroyAllWindows()
