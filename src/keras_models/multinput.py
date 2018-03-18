import cv2
import keras
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Model

from keras_models.base import AbstractNet


class MultiInputNeuralNet(AbstractNet):
    """
    Neutral network whose inputs are images, dlib points, dlib points distances from centroid point
    and dlib points vector angle with respect to centroid vector.

    Parameters
    ----------
    input_shape : tuple

    """

    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor, logger, session):
        super(MultiInputNeuralNet, self).__init__(data_out_dir, model_out_dir, input_shape, learning_rate, batch_size,
                                                  steps_per_epoch, epochs, preprocessor, logger, session)

        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (48,48,1)"

        self.TAG = "minn"
        self.feature_extractors = ["image"]
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        self.logger = super(MultiInputNeuralNet, self).init_logger(self.logger, self.model_out_dir, self.TAG)
        self.model = super(MultiInputNeuralNet, self).init_model(self.session)

    def build(self):
        """
        Build neural network model
        
        Returns 
        -------
        keras.models.Model : 
            neural network model
        """
        image_input_layer = Input(shape=self.input_shape)
        image_layer = Conv2D(32, (3, 3), activation='relu', padding="valid", kernel_initializer="glorot_normal")(
            image_input_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(64, (3, 3), activation="relu", padding="valid", kernel_initializer="glorot_normal")(
            image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(128, (3, 3), activation="relu", padding="valid", kernel_initializer="glorot_normal")(
            image_layer)
        image_layer = Flatten()(image_layer)

        dlib_points_input_layer = Input(shape=(1, 68, 2))
        dlib_points_layer = Conv2D(32, (1, 3), activation='relu', padding="valid", kernel_initializer="glorot_normal")(
            dlib_points_input_layer)
        dlib_points_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_layer)
        dlib_points_layer = Conv2D(64, (1, 3), activation="relu", padding="valid", kernel_initializer="glorot_normal")(
            dlib_points_layer)
        dlib_points_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_layer)
        dlib_points_layer = Conv2D(64, (1, 3), activation="relu", padding="valid", kernel_initializer="glorot_normal")(
            dlib_points_layer)

        dlib_points_layer = Flatten()(dlib_points_layer)

        dlib_points_dist_input_layer = Input(shape=(1, 68, 1))
        dlib_points_dist_layer = Conv2D(32, (1, 3), activation='relu', padding="valid",
                                        kernel_initializer="glorot_normal")(dlib_points_dist_input_layer)
        dlib_points_dist_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_dist_layer)
        dlib_points_dist_layer = Conv2D(64, (1, 3), activation="relu", padding="valid",
                                        kernel_initializer='glorot_normal')(dlib_points_dist_layer)
        dlib_points_dist_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_dist_layer)
        dlib_points_dist_layer = Conv2D(64, (1, 3), activation="relu", padding="valid",
                                        kernel_initializer='glorot_normal')(dlib_points_dist_layer)

        dlib_points_dist_layer = Flatten()(dlib_points_dist_layer)

        dlib_points_angle_input_layer = Input(shape=(1, 68, 1))
        dlib_points_angle_layer = Conv2D(32, (1, 3), activation='relu', padding="valid",
                                         kernel_initializer="glorot_normal")(dlib_points_angle_input_layer)
        dlib_points_angle_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_angle_layer)
        dlib_points_angle_layer = Conv2D(64, (1, 3), activation="relu", padding="valid",
                                         kernel_initializer='glorot_normal')(dlib_points_angle_layer)
        dlib_points_angle_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_angle_layer)
        dlib_points_angle_layer = Conv2D(64, (1, 3), activation="relu", padding="valid",
                                         kernel_initializer='glorot_normal')(dlib_points_angle_layer)

        dlib_points_angle_layer = Flatten()(dlib_points_angle_layer)

        merged_layers = keras.layers.concatenate(
            [image_layer, dlib_points_layer, dlib_points_dist_layer, dlib_points_angle_layer])

        merged_layers = Dense(252, activation='relu')(merged_layers)
        merged_layers = Dense(1024, activation='relu')(merged_layers)
        merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(self.number_of_class, activation='softmax')(merged_layers)

        self.model = Model(inputs=[image_input_layer, dlib_points_input_layer, dlib_points_dist_input_layer,
                                   dlib_points_angle_input_layer], outputs=merged_layers)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

        return self.model

    def train(self):
        assert self.model is not None, "Model not built yet."
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

        self.preprocessor = self.preprocessor(self.data_dir)

        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=([self.preprocessor.test_images, self.preprocessor.test_dpoints,
                                                   self.preprocessor.dpointsDists, self.preprocessor.dpointsAngles],
                                                  self.preprocessor.test_image_emotions))
        score = self.model.evaluate(
            [self.preprocessor.test_images, self.preprocessor.test_dpoints, self.preprocessor.dpointsDists,
             self.preprocessor.dpointsAngles], self.preprocessor.test_image_emotions)
        self.save_model()
        self.logger.log_model(self.TAG, score, self.model)

    def predict(self, face):
        """

        Args:
            face:

        Returns:

        """
        assert face.shape == IMG_SIZE, "Face image size should be " + str(IMG_SIZE)
        face = face.reshape(1, 64, 64)

        cv2.imshow("img", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        emotions = self.model.predict(face)[0]
        return emotions
