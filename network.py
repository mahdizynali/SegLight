# from config import *

# class Network(tf.keras.Model):
#     def __init__(self):
#         super(Network, self).__init__()

#         self.activation = tf.nn.leaky_relu
#         self.weightInitializer = tf.keras.initializers.GlorotUniform(seed=42)
#         self.normalizer_fn = None

#     def conv2d(self, input, nFilters, kernelSize, _strides, _name):
#         output = tf.keras.layers.Conv2D(
#             filters=nFilters,
#             kernel_size=kernelSize,
#             strides=_strides,
#             padding="SAME",
#             activation=self.activation,
#             kernel_initializer=self.weightInitializer,
#             name=_name
#         )(input)
#         return output

#     def sepConvMobileNet(self, features, kernel_size, out_filters, stride, _name, dilationFactor=1, pad='SAME'):
#         with tf.compat.v1.variable_scope(_name):
#             output = tf.keras.layers.SeparableConv2D(
#                 filters=out_filters,
#                 kernel_size=kernel_size,
#                 strides=stride,
#                 padding=pad,
#                 dilation_rate=dilationFactor,
#                 activation=self.activation,
#                 depth_multiplier=1,
#                 depthwise_initializer=self.weightInitializer,
#                 pointwise_initializer=self.weightInitializer,
#                 name='dw'
#             )(features)

#             output = tf.keras.layers.Conv2D(
#                 filters=out_filters,
#                 kernel_size=1,
#                 strides=1,
#                 padding='SAME',
#                 activation=self.activation,
#                 kernel_initializer=self.weightInitializer,
#                 name='pw'
#             )(output)
#             return output

#     def spp(self, input):
#         with tf.compat.v1.variable_scope("spp"):
#             prymid1 = self.sepConvMobileNet(input, 3, 24, 1, "p1", pad='SAME')
#             prymid2 = self.sepConvMobileNet(input, 3, 24, 1, "p2", pad='SAME')
#             prymid3 = self.sepConvMobileNet(input, 3, 24, 1, "p3", pad='SAME')
#             prymid4 = self.conv2d(input, 24, 1, 1, 'p4')

#             sppConcat = tf.concat([prymid1,prymid2,prymid3,prymid4],3,name="sppConcat")

#             return sppConcat

#     def model(self, input):
        
#         conv1 = self.conv2d(input, 6, 3, 2, 'conv1')

#         conv2 = self.sepConvMobileNet(conv1, 3, 12, 1, "conv2")
#         conv2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(conv2)

#         conv3 = self.sepConvMobileNet(conv2, 3, 12, 1, "conv3")
#         conv3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(conv3)

#         spp = self.spp(conv3)

#         pooling = self.conv2d(conv1, 12, 1, 1, 'pooling')

#         spp_merg = self.conv2d(spp, 48, 1, 1, 'spp-merg')
#         o1 = tf.image.resize(spp_merg, [120, 160], method=tf.image.ResizeMethod.BILINEAR)

#         concat = tf.concat([o1, pooling], 3, name="concat")

#         o2 = self.sepConvMobileNet(concat, 3, NUMBER_OF_CLASSES, 1, "o2", 1)

#         out = tf.image.resize(o2, [OUTPUT_HEIGHT, OUTPUT_WIDTH],method=tf.image.ResizeMethod.BILINEAR)

#         out = tf.keras.layers.Activation('softmax')(out)
        
#         return out



from config import *

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        self.activation = tf.nn.leaky_relu
        self.weight_initializer = tf.keras.initializers.GlorotUniform(seed=42)

    def conv2d(self, input, n_filters, kernel_size, strides, name):
        output = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            activation=self.activation,
            kernel_initializer=self.weight_initializer,
            name=name
        )(input)
        return output

    def sep_conv_mobilenet(self, features, kernel_size, out_filters, stride, name, dilation_factor=1, pad='SAME'):
        # Separable convolution with pointwise convolution
        output = tf.keras.layers.SeparableConv2D(
            filters=out_filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=pad,
            dilation_rate=dilation_factor,
            activation=self.activation,
            depth_multiplier=1,
            depthwise_initializer=self.weight_initializer,
            pointwise_initializer=self.weight_initializer,
            name=f"{name}_dw"
        )(features)

        output = tf.keras.layers.Conv2D(
            filters=out_filters,
            kernel_size=1,
            strides=1,
            padding='SAME',
            activation=self.activation,
            kernel_initializer=self.weight_initializer,
            name=f"{name}_pw"
        )(output)
        return output

    def spp(self, input):
        # Apply separate convolution layers with different dilation rates
        pyramid1 = self.sep_conv_mobilenet(input, 3, 24, 1, "p1", pad='SAME')
        pyramid2 = self.sep_conv_mobilenet(input, 3, 24, 1, "p2", pad='SAME')
        pyramid3 = self.sep_conv_mobilenet(input, 3, 24, 1, "p3", pad='SAME')
        pyramid4 = self.conv2d(input, 24, 1, 1, 'p4')

        spp_concat = tf.concat([pyramid1, pyramid2, pyramid3, pyramid4], axis=3, name="sppConcat")

        return spp_concat

    def call(self, inputs):
        conv1 = self.conv2d(inputs, 6, 3, 2, 'conv1')

        conv2 = self.sep_conv_mobilenet(conv1, 3, 12, 1, "conv2")
        conv2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(conv2)

        conv3 = self.sep_conv_mobilenet(conv2, 3, 12, 1, "conv3")
        conv3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(conv3)

        spp = self.spp(conv3)

        pooling = self.conv2d(conv1, 12, 1, 1, 'pooling')

        spp_merg = self.conv2d(spp, 48, 1, 1, 'spp-merg')
        o1 = tf.image.resize(spp_merg, [120, 160], method=tf.image.ResizeMethod.BILINEAR)

        concat = tf.concat([o1, pooling], axis=3, name="concat")

        o2 = self.sep_conv_mobilenet(concat, 3, NUMBER_OF_CLASSES, 1, "o2", 1)

        out = tf.image.resize(o2, [OUTPUT_HEIGHT, OUTPUT_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

        # Softmax activation at the end
        out = tf.keras.layers.Softmax()(out)
        
        return out
