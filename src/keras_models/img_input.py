import numpy as np
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential

from keras_models.base import AbstractNet


class ImageInputNeuralNet(AbstractNet):
    """
    """

    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor, logger, session):
        super(ImageInputNeuralNet, self).__init__(data_out_dir, model_out_dir, input_shape, learning_rate, batch_size,
                                                  steps_per_epoch, epochs, preprocessor, logger, session)

        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (48,48,1)"

        self.TAG = 'imnn'
        self.feature_extractors = ['image']
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        super(ImageInputNeuralNet, self).init_logger(self.logger, self.model_out_dir, self.TAG)
        super(ImageInputNeuralNet, self).init_model(self.session)

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
