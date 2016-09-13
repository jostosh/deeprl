from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class SpatialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(SpatialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape

        def rdim(out):
            return np.prod(input_shape) / input_shape[out]

        self.xmask = K.variable(np.linspace(-1, 1, input_shape[2]).reshape((1, 1, input_shape[2], 1)))
        self.ymask = K.variable(np.linspace(-1, 1, input_shape[3]).reshape((1, 1, 1, input_shape[3])))

        #self.xmask = np.stack(rdim(2) * [np.linspace(0, 1, input_shape[2])], axis=1).reshape(input_shape)
        #self.ymask = np.stack(rdim(3) * [np.linspace(0, 1, input_shape[3])], axis=0).reshape(input_shape)

    def call(self, x, mask=None):

        xsoft = K.exp(x) #* self.xmask
        xsoft = xsoft / K.sum(xsoft, axis=[2, 3], keepdims=True) # .reshape((-1, self.shape[1], 1, 1))  * self.xmask


        ysoft = K.exp(x) #* self.xmask
        ysoft = ysoft / K.sum(ysoft, axis=[2, 3], keepdims=True) # .reshape((-1, self.shape[1], 1, 1))  * self.xmask


        #ysoft = K.exp(x) * self.ymask

        output = K.concatenate([K.sum(xsoft, axis=[2, 3]), K.sum(ysoft, axis=[2, 3])], axis=1)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1] * 2)

