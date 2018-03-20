import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Layer


class Length(Layer):
    """
    """

    def call(self, inputs, **kwargs):
        """

        Args:
            inputs:
            kwargs:

        Returns:

        """
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        """

        Args:
            input_shape:

        Returns:

        """
        return input_shape[:-1]


class CapsLayer(Layer):
    """
    """

    def __init__(self, num_output=32,
                 batch_size=32, length_dim=8, num_caps=None, layer_type="pcap", num_rout_iter=3, **kwargs):
        """
        Args:
            num_caps: number capsules in this layer
            length_dim: dimension of capsules output length
            layer_type: type of layer either primary capsule layer(pcap) or capsule layer(cap).
        """
        super(CapsLayer, self).__init__(**kwargs)
        self.num_output = num_output
        self.length_dim = length_dim
        self.layer_type = layer_type
        self.num_rout_iter = num_rout_iter
        self.num_caps = num_caps

    def call(self, input, kernel_size=None, strides=2, padding="valid"):
        """

        Args:
            input:
            kernel_size:
            strides:
            padding:

        Returns:

        """
        if kernel_size is None:
            kernel_size = [9, 9]
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        if self.layer_type == "pcap":
            capsules = []
            for i in range(self.num_output):
                caps_i = Conv2D(self.length_dim, kernel_size=self.kernel_size, strides=self.strides,
                                activation="relu", padding=self.padding, name="conv_" + str(i))(input)
                caps_i_shape = caps_i.shape.as_list()
                caps_i = K.reshape(caps_i, (-1, caps_i_shape[1] * caps_i_shape[2], self.length_dim))
                capsules.append(caps_i)
            capsules_shape = capsules[0].shape.as_list()
            print(capsules_shape, "primary caps")
            self.num_caps = capsules_shape[1] * self.num_output
            capsules = keras.layers.concatenate(capsules, axis=1)
            return capsules

        elif self.layer_type == "cap":
            # input.shape (-1,cpa)
            self.net_input = input
            caps = self.routing(self.net_input)
            return caps
        else:
            raise Exception("Not implemented for " + str(self.layer_type))

    def routing(self, input):
        """

        Args:
            input:

        Returns:

        """

        # input shape None,num_caps,input_length_dim
        input = K.expand_dims(input, axis=2)
        input = K.expand_dims(input, axis=3)
        # None,input_num_caps,1,1,input_length_dim
        input = K.tile(input, [1, 1, self.num_caps, 1, 1])
        # input shape (?, input_num_caps,self.num_caps,1,input_length_dim)
        # weight shape (32, 32, 6, 6,10,8,16)\
        print(input.shape)
        input_shape = input.shape.as_list()
        weight_shape = [input_shape[1], self.num_caps, input_shape[4], self.length_dim]
        W = self.add_weight(shape=weight_shape,
                            initializer='glorot_uniform',
                            name='W')
        b_IJ = self.add_weight(shape=[1, input_shape[1], self.num_caps, 1, 1],
                               initializer="zeros",
                               name='bias',
                               trainable=False
                               )

        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, W, [3, 2]),
                             elems=input,
                             initializer=K.zeros([input_shape[1], self.num_caps, 1, self.length_dim]))

        print("uhat ", inputs_hat.shape)
        # print "uhat", u_hat.shape
        for iter in range(self.num_rout_iter):
            # b_IJ shape b_J (1, 1152, 10, 1, 1)
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            s_J = K.sum(c_IJ * inputs_hat, 1, keepdims=True)
            v_J = self.squash(s_J)
            if iter != self.num_rout_iter - 1:
                b_IJ += K.sum(inputs_hat * v_J, -1, keepdims=True)

        v_J = K.reshape(v_J, [-1, self.num_caps, self.length_dim])
        return v_J

    @staticmethod
    def squash(vector):
        """

        Args:
            vector:

        Returns:

        """
        vec_squared_norm = K.sum(K.square(vector), axis=-1, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / K.sqrt(vec_squared_norm)
        vec_squashed = scalar_factor * vector  # element-wise
        return vec_squashed

    def compute_output_shape(self, input_shape):
        """

        Args:
            input_shape:

        Returns:

        """
        return tuple([None, self.num_caps, self.length_dim])
