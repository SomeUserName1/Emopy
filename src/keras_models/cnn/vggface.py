# coding=utf-8
from keras import Model, losses, optimizers
from keras.layers import Dense, Input, PReLU, Dropout, Conv2D, Concatenate
from keras_vggface.vggface import VGGFace

from keras_models.cnn.multinput import MultiInputNeuralNet


class VGGFaceEmopyNet(MultiInputNeuralNet):
    """
    Class for implementation of EmoPy using VGG Face Net as base
    according to http://www.robots.ox.ac.uk/%7Evgg/software/vgg_face/.
    """

    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs,
                 preprocessor, logger, session):
        super(VGGFaceEmopyNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, learning_rate,
                                              batch_size,
                                              steps_per_epoch, epochs, preprocessor, logger, session)

        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (64,64,1)"

        self.feature_extractors = ["image"]
        self.number_of_classes = self.preprocessor.classifier.get_num_class()
        self.model = super(VGGFaceEmopyNet, self).init_model(session)

    def build(self):
        """

        Returns:
            An instance of the EmoPy VGG Face model
        """
        # x = VGGFace(include_top=False, input_shape=self.input_shape)
        image_input = Input(shape=self.input_shape)

        vgg_out = VGGFace(model='vgg16', include_top=False, weights=None, classes=7, input_shape=self.input_shape,
                          pooling='avg')(image_input)
        vgg_out = Dense(4048)(vgg_out)
        vgg_out = PReLU()(vgg_out)
        vgg_out = Dropout(0.382)(vgg_out)
        vgg_out = Dense(1024)(vgg_out)
        vgg_out = PReLU()(vgg_out)
        vgg_out = Dense(256)(vgg_out)
        vgg_out = PReLU()(vgg_out)

        dlib_points_input_layer = Input(shape=(1, 68, 2))
        dlib_points_angle_input_layer = Input(shape=(1, 68, 1))
        dlib_points_dist_input_layer = Input(shape=(1, 68, 1))
        dlib_stack = Concatenate()([dlib_points_input_layer, dlib_points_dist_input_layer, dlib_points_input_layer])
        dlib_stack = Conv2D(32, (1, 3), padding="valid", kernel_initializer="glorot_normal")(dlib_stack)
        dlib_stack = Dense(256)(dlib_stack)
        dlib_stack = PReLU()(dlib_stack)

        merged_layers = Concatenate()([vgg_out, dlib_stack])
        merged_layers = Dense(512)(merged_layers)
        merged_layers = PReLU()(merged_layers)
        merged_layers = Dense(self.number_of_classes, activation='softmax')(merged_layers)

        self.model = Model(inputs=[image_input, dlib_points_input_layer, dlib_points_dist_input_layer,
                                   dlib_points_angle_input_layer], outputs=merged_layers)

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])
        return self.model
