import os

import keras
import numpy as np
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_json

from keras_models.base import AbstractNet


class NeuralNet(AbstractNet):
    """
    Base class for all neural keras_models.

    Parameters
    ----------
    input_shape : tuple

    """

    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor, logger, session):

        super(NeuralNet, self).__init__(data_out_dir, model_out_dir, input_shape, learning_rate, batch_size,
                                        steps_per_epoch, epochs, preprocessor, logger, session)

        self.TAG = 'imnn'
        super(NeuralNet, self).init_logger(self.logger, self.model_out_dir, self.TAG)

        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (48,48,1)"
        self.feature_extractors = ['image']
        self.number_of_class = self.preprocessor.classifier.get_num_class()

        if session == 'train':
            self.model = self.build()
        else:
            self.model = self.load_model(model_out_dir)

    def build(self):
        """
        Build neural network model

        Returns
        -------
        keras.models.Model :
            neural network model
        """
        # TODO rework, use capsule impl., PReLU, BN
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=self.input_shape,
                         kernel_initializer="glorot_normal"))
        # model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer="glorot_normal"))
        # model.add(Dropout(0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer="glorot_normal"))
        # model.add(Dropout(0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer="glorot_normal"))
        # model.add(Dropout(0.2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(252, (3, 3), activation='relu',padding= "same",kernel_initializer="glorot_normal"))
        # model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # TODO REWORK DENSE LAYERS
        model.add(Dense(252, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.number_of_class, activation='softmax'))

        return model

    def train(self):
        """Traines the neuralnet model.
        This method requires the following two directory to exist
        /PATH-TO-DATASET-DIR/train
        /PATH-TO-DATASET-DIR/test

        """

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])
        # self.model.fit(x_train,y_train,epochs = EPOCHS,
        #                 batch_size = BATCH_SIZE,validation_data=(x_test,y_test))
        self.preprocessor = self.preprocessor(DATA_SET_DIR)
        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=(self.preprocessor.test_images, self.preprocessor.test_image_emotions))
        score = self.model.evaluate(self.preprocessor.test_images, self.preprocessor.test_image_emotions)
        self.save_model()
        self.logger.log_model(self.models_local_folder, score)

    def save_model(self):
        """
        Saves NeuralNet model. The naming convention is for json and h5 files is,
        `/path-to-models/model-local-folder-model-number.json` and
        `/path-to-models/model-local-folder-model-number.h5` respectively.
        This method also increments model_number inside "model_number.txt" file.
        """

        if not os.path.exists(PATH2SAVE_MODELS):
            os.makedirs(PATH2SAVE_MODELS)
        if not os.path.exists(os.path.join(PATH2SAVE_MODELS, self.models_local_folder)):
            os.makedirs(os.path.join(PATH2SAVE_MODELS, self.models_local_folder))
        if not os.path.exists(os.path.join(PATH2SAVE_MODELS, self.models_local_folder, "model_number.txt")):
            model_number = np.array([0])
        else:
            model_number = np.fromfile(os.path.join(PATH2SAVE_MODELS, self.models_local_folder, "model_number.txt"),
                                       dtype=int)
        model_file_name = self.models_local_folder + "-" + str(model_number[0])
        with open(os.path.join(PATH2SAVE_MODELS, self.models_local_folder, model_file_name + ".json"), "a+") as jfile:
            jfile.write(self.model.to_json())
        self.model.save_weights(os.path.join(PATH2SAVE_MODELS, self.models_local_folder, model_file_name + ".h5"))
        model_number[0] += 1
        model_number.tofile(os.path.join(PATH2SAVE_MODELS, self.models_local_folder, "model_number.txt"))

    def load_model(self, model_path):
        """

        Args:
            model_path:

        Returns:

        """
        with open(model_path + ".json") as model_file:
            model = model_from_json(model_file.read())
            model.load_weights(model_path + ".h5")
            return model

    def predict(self, face):
        """

        Args:
            face:

        Returns:

        """
        assert face.shape == IMG_SIZE, "Face image size should be " + str(IMG_SIZE)
        face = face.reshape(-1, 64, 64, 1)
        face = face.astype(np.float32) / 255
        emotions = self.model.predict(face)
        return emotions
