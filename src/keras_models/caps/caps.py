import keras
from keras import backend as K
from keras.layers import Conv2D, Input, PReLU, BatchNormalization
from keras.models import Model

from keras_models.base import AbstractNet
from keras_models.caps.layers import Length, CapsLayer


class CapsNet(AbstractNet):
    """
    """

    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                 preprocessor, logger, session, lmd):
        """

        Args:
            input_shape:
            lmd:
        """
        super(CapsNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                                      preprocessor, logger, session)

        self.lmd = lmd
        self.feature_extractors = ["image"]
        self.number_of_classes = self.preprocessor.classifier.get_num_class()
        self.model = super(CapsNet, self).init_model(self.session)

    def build(self):
        input_layer = Input(shape=self.input_shape)

        conv1 = Conv2D(32, kernel_size=[9, 9], strides=1, padding="valid", name="conv1")(input_layer)
        conv1 = PReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
        primary_caps = CapsLayer(length_dim=8)(conv1, padding="valid")
        second_caps = CapsLayer(num_caps=self.number_of_classes, length_dim=16, layer_type="cap")(primary_caps)
        length = Length(name="pred")(second_caps)

        self.model = Model(inputs=input_layer, outputs=length)
        self.model.compile(loss=[self.margin_loss],
                           optimizer=keras.optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

    def train(self):
        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=(
                                     self.preprocessor.test_images, self.preprocessor.test_image_emotions))
        score = self.model.evaluate(self.preprocessor.test_images, self.preprocessor.test_image_emotions)

        self.save_model()
        self.logger.log_model(self.net_type, score, self.model)

    def margin_loss(self, y_true, y_pred):
        """

        Args:
            y_true:
            y_pred:

        Returns:

        """

        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            self.lmd * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

    def predict(self, face):
        pass
