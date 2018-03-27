import keras
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, PReLU, Dropout, Flatten
from keras.models import Model

from keras_models.base import AbstractNet


class ImageInputNeuralNet(AbstractNet):
    """
    """

    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs,
                 preprocessor, logger, session):
        super(ImageInputNeuralNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, learning_rate,
                                                  batch_size,
                                                  steps_per_epoch, epochs, preprocessor, logger, session)

        self.feature_extractors = ['image']
        self.number_of_classes = self.preprocessor.classifier.get_num_class()
        self.model = super(ImageInputNeuralNet, self).init_model(self.session)

    def build(self):
        """
        Build neural network model

        Returns
        -------
        keras.models.Model :
            neural network model
        """
        image_input_layer = Input(shape=self.input_shape)
        image_layer = BatchNormalization()(image_input_layer)
        image_layer = Conv2D(32, (3, 3), padding="valid", kernel_initializer="glorot_normal")(
            image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = BatchNormalization()(image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(32, (1, 1), padding="valid", kernel_initializer="glorot_normal") \
            (image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = BatchNormalization()(image_layer)
        image_layer = Conv2D(64, (3, 3), padding="valid", kernel_initializer="glorot_normal")(
            image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = BatchNormalization()(image_layer)
        image_layer = Conv2D(128, (3, 3), padding="valid", kernel_initializer="glorot_normal")(image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = BatchNormalization()(image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(256, (3, 3), padding="valid", kernel_initializer="glorot_normal")(image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = BatchNormalization()(image_layer)
        image_layer = Flatten()(image_layer)
        image_layer = Dense(256)(image_layer)
        image_layer = Dense(1024)(image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = Dense(2048)(image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = Dropout(0.25)(image_layer)
        image_layer = Dense(512)(image_layer)
        image_layer = PReLU()(image_layer)
        image_layer = Dense(self.number_of_classes, activation='softmax')(image_layer)

        self.model = Model(inputs=image_input_layer, outputs=image_layer)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

        return self.model

    def train(self):
        self.preprocessor = self.preprocessor(self.data_dir)
        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=(self.preprocessor.test_images, self.preprocessor.test_image_emotions))
        score = self.model.evaluate(self.preprocessor.test_images, self.preprocessor.test_image_emotions)

        self.save_model()
        self.logger.log_model(self.net_type, score, self.model)

    def predict(self, face):
        """

        Args:
            face:

        Returns:

        """
        assert face.shape == self.input_shape, "Face image size should be " + str(self.input_shape)
        face = face.reshape(-1, 64, 64, 1)
        face = face.astype(np.float32) / 255
        emotions = self.model.predict(face)
        return emotions
