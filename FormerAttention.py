from keras.layers import *
from keras.models import *
from keras import backend as K

#defining FormerAttention layer
class FormerAttention(Layer):
    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(FormerAttention,self).__init__(name=name)
        self.return_sequences = return_sequences
        super(FormerAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")

        super(FormerAttention,self).build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
