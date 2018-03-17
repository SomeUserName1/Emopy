import os
from abc import abstractmethod, ABCMeta

import keras
import numpy as np
from keras.models import model_from_json

from preprocess.dlib_input import DlibInputPreprocessor
from preprocess.image_input import Preprocessor
from preprocess.multinput import MultiInputPreprocessor
from preprocess.sequencial import SequencialPreprocessor, DlibSequencialPreprocessor
from util.BaseLogger import EmopyLogger


class AbstractNet(object, metaclass=ABCMeta):
    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor=None, logger=None, session='train'):
        """
        initializes the basic class variables and the non-basic (e.g. different preprocessors) to None
        It is important to set the TAG of the net directly after calling super.init and esp. before initializing the
        logger
        Args:
            data_out_dir: directory where the data_collectors outputted to
            model_out_dir: directory where the weights, all logs and eventually visualizations of the weights are saved
            input_shape: the shape (width & height) of the input images
            learning_rate: the chosen learning rate
            batch_size: the amount of items per batch
            steps_per_epoch: the amounts of batches per epoch
            epochs: the amount of epochs
            preprocessor: A dedicated preprocessor to be set after calling super.init
            logger: The standard logger found int util/BaseLogger.py if None; by now there are no dedicated loggers
            session: either
        """
        self.TAG = 'an'
        self.session = session
        self.logger = logger
        self.data_dir = data_out_dir
        self.model_out_dir = model_out_dir

        self.input_shape = input_shape
        self.preprocessor = preprocessor
        self.number_of_classes = None

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = None

    def init_logger(self, logger, model_out_dir, tag):
        """
        initializes the logger when the TAG is set according to the net type
        :param logger: the used logger
        :param model_out_dir: the dir where the logger shall write to
        :param tag: the tag which is set according to the net type
        """
        if not os.path.exists(os.path.join(model_out_dir, tag)):
            os.makedirs(os.path.join(model_out_dir, tag))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(model_out_dir, tag, "%s.log" % tag)])
        else:
            self.logger = logger

    def init_model(self, session):
        if session == 'train':
            self.model = self.build()
        elif session == 'predict':
            self.model = self.load_model()

    @abstractmethod
    def build(self):
        """
        Build neural network model

        Returns
        -------
        keras.models.Model :
            neural network model
        """
        pass

    def train(self):
        """


        """
        assert self.model is not None, "Model not built yet."

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])
        self.preprocessor = self.preprocessor(self.data_dir)
        self.model.summary()

        if isinstance(self.preprocessor, Preprocessor):
            validation_data = self.preprocessor.test_images, \
                              self.preprocessor.test_image_emotions

        elif isinstance(self.preprocessor, DlibInputPreprocessor):
            validation_data = [self.preprocessor.test_dpoints, self.preprocessor.dpointsDists,
                               self.preprocessor.dpointsAngles], self.preprocessor.test_image_emotions

        elif isinstance(self.preprocessor, MultiInputPreprocessor):
            validation_data = [self.preprocessor.test_images, self.preprocessor.test_dpoints,
                               self.preprocessor.dpointsDists, self.preprocessor.dpointsAngles], \
                              self.preprocessor.test_image_emotions

        elif isinstance(self.preprocessor, SequencialPreprocessor):
            validation_data = self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels

        elif isinstance(self.preprocessor, DlibSequencialPreprocessor):
            validation_data = self.preprocessor.test_sequences_dpoints, self.preprocessor.test_sequence_labels
        else:
            raise Exception("invalid preprocessor %s" % self.preprocessor)

        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=validation_data)
        score = self.model.evaluate(validation_data)
        self.save_model()
        self.logger.log_model(self.TAG, score)

    def save_model(self):
        """
        Saves NeuralNet model. The naming convention is for json and h5 files is,
        `/path-to-models/model-local-folder-model-number.json` and
        `/path-to-models/model-local-folder-model-number.h5` respectively.
        This method also increments model_number inside "model_number.txt" file.
        """
        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)
        out_dir = os.path.join(self.model_out_dir, self.TAG)
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir))
        if not os.path.exists(os.path.join(out_dir, "model_number.txt")):
            model_number = np.array([0])
        else:
            model_number = np.fromfile(os.path.join(out_dir, "model_number.txt"),
                                       dtype=int)
        model_file_name = self.TAG + "-" + str(model_number[0])
        with open(os.path.join(self.model_out_dir, self.TAG, model_file_name + ".json"), "a+") as jfile:
            jfile.write(self.model.to_json())
        self.model.save_weights(os.path.join(out_dir, model_file_name + ".h5"))
        model_number[0] += 1
        model_number.tofile(os.path.join(out_dir, "model_number.txt"))

    def load_model(self):
        """
        loads a pre-trained model
        Args:
            model_path: path to a pre-trained model without file-ending

        Returns: the loaded model

        """
        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)
        out_dir = os.path.join(self.model_out_dir, self.TAG)
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir))
        if not os.path.exists(os.path.join(out_dir, "model_number.txt")):
            raise Exception("no models created by now, run training first")
        else:
            model_number = np.fromfile(os.path.join(out_dir, "model_number.txt"),
                                       dtype=int)

        model_file_name = self.TAG + "-" + str(model_number[0])
        path = os.path.join(self.model_out_dir, self.TAG, model_file_name + ".json")

        with open(path) as model_file:
            model = model_from_json(model_file.read())
            model.load_weights(path + ".h5")
            return model

    @abstractmethod
    def predict(self, face):
        """

        """
        pass
