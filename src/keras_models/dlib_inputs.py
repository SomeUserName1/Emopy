import keras
from keras.layers import Input, Flatten, Dense, Conv2D, Dropout
from keras.models import Model

from keras_models.base import AbstractNet


class DlibPointsInputNeuralNet(AbstractNet):
    """
    Neutral network whose inputs are dlib points, dlib points distances from centroid point
    and dlib points vector angle with respect to centroid vector.
    """

    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor, logger, session):
        """

        Args:
            input_shape:
            preprocessor:
            logger:
            train:
        """
        super(DlibPointsInputNeuralNet, self).__init__(data_out_dir, model_out_dir, input_shape, learning_rate,
                                                       batch_size, steps_per_epoch, epochs, preprocessor, logger,
                                                       session)
        self.TAG = 'dlibnn'
        self.feature_extractors = ["dlib"]
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        super(DlibPointsInputNeuralNet, self).init_logger(self.logger, self.model_out_dir, self.TAG)
        super(DlibPointsInputNeuralNet, self).init_model(self.session)

    def build(self):
        """
        Build neural network model
        
        Returns 
        -------
        keras.models.Model : 
            neural network model
        """

        dlib_points_input_layer = Input(shape=(1, 68, 2))
        dlib_points_layer = Conv2D(32, (1, 3), activation='relu', padding="same", kernel_initializer="glorot_normal")(
            dlib_points_input_layer)
        dlib_points_layer = Conv2D(64, (1, 3), activation="relu", padding="same", kernel_initializer="glorot_normal")(
            dlib_points_layer)
        # dlib_points_layer = Conv2D(128,(1, 3),activation = "relu",padding="same",kernel_initializer="glorot_normal")(dlib_points_layer)

        dlib_points_layer = Flatten()(dlib_points_layer)

        dlib_points_dist_input_layer = Input(shape=(1, 68, 1))
        dlib_points_dist_layer = Conv2D(32, (1, 3), activation='relu', padding="same",
                                        kernel_initializer="glorot_normal")(dlib_points_dist_input_layer)
        dlib_points_dist_layer = Conv2D(64, (1, 3), activation="relu", padding="same",
                                        kernel_initializer='glorot_normal')(dlib_points_dist_layer)
        # dlib_points_dist_layer = Conv2D(128,(1, 3),activation = "relu",padding="same",kernel_initializer='glorot_normal')(dlib_points_dist_layer)

        dlib_points_dist_layer = Flatten()(dlib_points_dist_layer)

        dlib_points_angle_input_layer = Input(shape=(1, 68, 1))
        dlib_points_angle_layer = Conv2D(32, (1, 3), activation='relu', padding="same",
                                         kernel_initializer="glorot_normal")(dlib_points_angle_input_layer)
        dlib_points_angle_layer = Conv2D(64, (1, 3), activation="relu", padding="same",
                                         kernel_initializer='glorot_normal')(dlib_points_angle_layer)
        # dlib_points_angle_layer = Conv2D(18,(1, 3),activation = "relu",padding="same",kernel_initializer='glorot_normal')(dlib_points_angle_layer)

        dlib_points_angle_layer = Flatten()(dlib_points_angle_layer)

        merged_layers = keras.layers.concatenate([dlib_points_layer, dlib_points_dist_layer, dlib_points_angle_layer])

        merged_layers = Dense(128, activation='relu')(merged_layers)
        # merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(1024, activation='relu')(merged_layers)
        merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(self.number_of_class, activation='softmax')(merged_layers)

        self.model = Model(
            inputs=[dlib_points_input_layer, dlib_points_dist_input_layer, dlib_points_angle_input_layer],
            outputs=merged_layers)
        self.built = True
        return self.model

    def train(self):
        assert self.model is not None, "Model not built yet."
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

        self.preprocessor = self.preprocessor(self.data_dir)

        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=([self.preprocessor.test_dpoints, self.preprocessor.dpointsDists,
                                                   self.preprocessor.dpointsAngles],
                                                  self.preprocessor.test_image_emotions))
        score = self.model.evaluate(
            [self.preprocessor.test_dpoints, self.preprocessor.dpointsDists, self.preprocessor.dpointsAngles],
            self.preprocessor.test_image_emotions)
        self.save_model()
        self.logger.log_model(self.TAG, score, self.model)

    def predict(self, face):
        """

        Args:
            face:

        Returns:

        """
        assert face.shape == IMG_SIZE, "Face image size should be " + str(IMG_SIZE)
        face = face.reshape(-1, 48, 48, 1)
        emotions = self.model.predict(face)[0]
        return emotions
