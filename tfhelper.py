from math import ceil
import tensorflow as tf

class TFHelper:


    def __init__(self, strides, dropout, shapes, pipeline):
        self.strides = strides
        self.dropout = dropout
        self.shapes = shapes
        self.pipeline = pipeline




    def shape(self, input, output):
        #print("shape " + input + ", " + output)
        height = 1+  ceil(float(self.shapes[input][0] - self.shapes[output][0] ) / float(self.strides[output]))
        width =  1+ ceil((self.shapes[input][1] - self.shapes[output][1] ) / float(self.strides[output]))
        #print("shape "+input+", "+output+" = "+str((height, width, shapes[input][-1], shapes[output][-1])))
        return [height, width, self.shapes[input][-1], self.shapes[output][-1]]

    def inputOf(self, layer):
        i = self.pipeline.index(layer)
        if(i > 0):
            return self.pipeline[i-1]


    def randNormVar(self, param):
        #print("making random var of size "+str(param))
        return tf.Variable(tf.truncated_normal(param))


    def bias(self, idLayer):
            return self.randNormVar([self.shapes[idLayer][-1]])


    def weight(self, idLayer):
        return self.randNormVar(self.shape(self.inputOf(idLayer), idLayer))


    def stride(self, idLayer):
        return [1, self.strides[idLayer], self.strides[idLayer], 1]

    # inputs : A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
    # new_height = (input_height - filter_height)/ S + 1
    # new_width = (input_width - filter_width)/ S + 1
    def poolLayer(self, inputs, idLayer, padding='VALID'):
        layer = tf.nn.max_pool(inputs, ksize=self.stride(idLayer), strides=self.stride(idLayer), padding=padding, name=idLayer)
        return layer

    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # new_height = (input_height - filter_height + 2 * P)/S + 1
    # new_width = (input_width - filter_width + 2 * P)/S + 1
    def convLayer(self, inputs, idLayer, padding='VALID'):
        layer = tf.nn.conv2d(inputs, self.weight(idLayer), strides=self.stride(idLayer), padding=padding)
        layer = tf.nn.bias_add(layer, self.bias(idLayer))
        return tf.nn.relu(layer, name=idLayer)


    def fullLayer(self, inputs, idLayer):
        print(idLayer)

        num_hidden_neurons = self.shapes[idLayer][0]
        incoming = self.shapes[self.inputOf(idLayer)][0]

        hidden_weights = self.randNormVar([incoming, num_hidden_neurons])
        hidden_biases = self.randNormVar([num_hidden_neurons])
        # now the layer itself. It multiplies data by weights, adds biases
        # and takes ReLU over result
        hidden_layer = tf.nn.relu(tf.matmul(inputs, hidden_weights) + hidden_biases)

        drop = tf.nn.dropout(hidden_layer, tf.constant(self.dropout[idLayer]))
        return drop

        #return tf.contrib.layers.fully_connected(inputs, self.shapes[idLayer][0], tf.nn.relu)
