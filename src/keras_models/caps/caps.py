import keras
import numpy as np
from keras import backend as K
from keras import callbacks as cb
from keras.layers import Conv2D, Input, Dense, Reshape, Add
from keras.models import Model, Sequential

from keras_models.base import AbstractNet
from keras_models.caps.layers import Length, CapsuleLayer, PrimaryCap, Mask

K.set_image_data_format('channels_last')


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
        self.lr_decay = 0.9
        self.model = super(CapsNet, self).init_model(self.session)

    def build(self):

        input_layer = Input(shape=self.input_shape)

        conv1 = Conv2D(256, kernel_size=[9, 9], strides=1, activation='relu', padding="valid", name="conv1")(
            input_layer)

        primary_caps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

        emotion_caps = CapsuleLayer(num_capsule=self.number_of_classes, dim_capsule=16)(primary_caps)
        out_caps = Length(name="pred")(emotion_caps)

        # Decoder network.
        y = Input(shape=(self.number_of_classes,))
        masked_by_y = Mask()(
            [emotion_caps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(emotion_caps)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = Sequential(name='decoder')
        decoder.add(Dense(512, activation='relu', input_dim=16 * self.number_of_classes))
        decoder.add(Dense(1024, activation='relu'))
        decoder.add(Dense(np.prod(self.input_shape), activation='sigmoid'))
        decoder.add(Reshape(target_shape=self.input_shape, name='out_recon'))

        # Models for training and evaluation (prediction)
        train_model = Model([input_layer, y], [out_caps, decoder(masked_by_y)])
        eval_model = Model(input_layer, [out_caps, decoder(masked)])

        # manipulate model
        noise = Input(shape=(self.number_of_classes, 16))
        noised_digitcaps = Add()([emotion_caps, noise])
        masked_noised_y = Mask()([noised_digitcaps, y])
        manipulate_model = Model([input_layer, y, noise], decoder(masked_noised_y))

        train_model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                            loss=[keras.losses.categorical_crossentropy, 'mse'],
                            loss_weights=[1., 0.2727272],
                            metrics=['accuracy'])

        return train_model  # , eval_model, manipulate_model

    def train(self):
        dir = self.model_out_dir + '/' + self.net_type + '/'
        tb = cb.TensorBoard(log_dir=dir + '/tensorboard-logs',
                            batch_size=self.batch_size, histogram_freq=int(10))
        checkpoint = cb.ModelCheckpoint(dir + '/weights-{epoch:02d}.h5', monitor='pred_acc',
                                        save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = cb.LearningRateScheduler(schedule=lambda epoch: self.learning_rate * (self.lr_decay ** epoch))

        self.preprocessor = self.preprocessor(self.data_dir)
        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=(
                                     [self.preprocessor.test_images, self.preprocessor.test_image_emotions],
                                     [self.preprocessor.test_image_emotions, self.preprocessor.test_images]),
                                 callbacks=[tb, checkpoint, lr_decay])
        score = self.model.evaluate([self.preprocessor.test_images, self.preprocessor.test_image_emotions],
                                    [self.preprocessor.test_image_emotions, self.preprocessor.test_images])

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
        assert face.shape == self.input_shape[:-1], "Face image size should be " + str(self.input_shape[:-1])
        face = face.reshape(-1, 64, 64, 1)
        face = face.astype(np.float32) / 255
        emotions = self.model.predict(face)
        return emotions
