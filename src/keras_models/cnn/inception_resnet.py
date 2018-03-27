from keras import losses, optimizers
from keras.applications.inception_resnet_v2 import *
from keras.layers import Dropout
from keras.layers.advanced_activations import PReLU

from keras_models.cnn.multinput import MultiInputNeuralNet


class InceptionResNet(MultiInputNeuralNet):
    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs,
                 preprocessor, logger, session):
        super(InceptionResNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, learning_rate,
                                              batch_size,
                                              steps_per_epoch, epochs, preprocessor, logger, session)

        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (64,64,1)"

        self.feature_extractors = ["image"]
        self.number_of_classes = self.preprocessor.classifier.get_num_class()
        self.model = super(InceptionResNet, self).init_model(session)

    def build(self):
        input_tensor = Input(shape=self.input_shape)
        inc_res_net = InceptionResNetV2(include_top=False, weights=None, classes=7, input_shape=self.input_shape,
                                        pooling='avg', input_tensor=input_tensor)
        res_out = inc_res_net(input_tensor)
        x = Dense(4096)(res_out)
        x = PReLU()(x)
        x = Dropout(0.382)(x)
        x = Dense(4096)(x)
        x = PReLU()(x)
        x = Dense(2048)(x)
        x = PReLU()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        x = Dense(self.number_of_classes, activation='softmax', name='predictions')(x)
        self.model = Model(input_tensor, x)

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])
        return self.model
