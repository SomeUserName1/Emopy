# coding=utf-8
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace

from keras_models.base import AbstractNet


# TODO test
class VGGFaceEmopyNet(AbstractNet):
    """
    Class for implementation of EmoPy using VGG Face Net as base
    according to http://www.robots.ox.ac.uk/%7Evgg/software/vgg_face/.
    """

    def predict(self, faces):
        pass

    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps, epochs,
                 preprocessor, logger, session):
        super(VGGFaceEmopyNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, lr, batch_size, steps,
                                              epochs, preprocessor, logger, session)
        self.TAG = "vgg"
        self.max_sequence_length = 10
        self.feature_extractors = ['image']
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        super(VGGFaceEmopyNet, self).init_model(self.session)

    def train(self):
        pass

    def build(self):
        """

        Returns:
            An instance of the EmoPy VGG Face model
        """
        # x = VGGFace(include_top=False, input_shape=self.input_shape)
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(32, activation='relu', name='fc6')(x)
        x = Dense(32, activation='relu', name='fc7')(x)
        print("VGG")
        x.summary()
        return vgg_model
