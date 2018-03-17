import os
from abc import abstractmethod, ABCMeta

from util.BaseLogger import EmopyLogger


class AbstractNet(object, metaclass=ABCMeta):
    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor=None, logger=None, session='train'):
        """

        Args:
            data_out_dir:
             model_out_dir:
            input_shape:
            learning_rate:
            batch_size:
            steps_per_epoch:
            epochs:
            preprocessor:
            logger:
            session:
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

    @abstractmethod
    def train(self):
        """Traines the neuralnet model.
        This method requires the following two directory to exist
        /PATH-TO-DATASET-DIR/train
        /PATH-TO-DATASET-DIR/test

        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Saves NeuralNet model. The naming convention is for json and h5 files is,
        `/path-to-models/model-local-folder-model-number.json` and
        `/path-to-models/model-local-folder-model-number.h5` respectively.
        This method also increments model_number inside "model_number.txt" file.
        """
        pass

    @abstractmethod
    def load_model(self, model_path):
        """

        Args:
            model_path:

        Returns:

        """
        pass

    @abstractmethod
    def evaluate(self):
        """

        """
        pass

    @abstractmethod
    def predict(self, face):
        """

        """
        pass

    def init_logger(self, logger, model_out_dir, tag):
        if not os.path.exists(os.path.join(model_out_dir, tag)):
            os.makedirs(os.path.join(model_out_dir, tag))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(model_out_dir, tag, "%s.txt" % tag)])
        else:
            self.logger = logger
