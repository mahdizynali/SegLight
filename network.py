from config import *

# class Network(tf.keras.Model):
#     def __init__(self):
#         super(Network, self).__init__()

#         self.activation = tf.nn.leaky_relu
#         self.weight_initializer = tf.keras.initializers.GlorotUniform(seed=42)

#     def conv2d(self, input, n_filters, kernel_size, strides, name):
#         output = tf.keras.layers.Conv2D(
#             filters=n_filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding="SAME",
#             activation=self.activation,
#             kernel_initializer=self.weight_initializer,
#             name=name
#         )(input)
#         return output

#     def sep_conv_mobilenet(self, features, kernel_size, out_filters, stride, name, dilation_factor=1, pad='SAME'):
#         # Separable convolution with pointwise convolution
#         output = tf.keras.layers.SeparableConv2D(
#             filters=out_filters,
#             kernel_size=kernel_size,
#             strides=stride,
#             padding=pad,
#             dilation_rate=dilation_factor,
#             activation=self.activation,
#             depth_multiplier=1,
#             depthwise_initializer=self.weight_initializer,
#             pointwise_initializer=self.weight_initializer,
#             name=f"{name}_dw"
#         )(features)

#         output = tf.keras.layers.Conv2D(
#             filters=out_filters,
#             kernel_size=1,
#             strides=1,
#             padding='SAME',
#             activation=self.activation,
#             kernel_initializer=self.weight_initializer,
#             name=f"{name}_pw"
#         )(output)
#         return output

#     def spp(self, input):
#         # Apply separate convolution layers with different dilation rates
#         pyramid1 = self.sep_conv_mobilenet(input, 3, 24, 1, "p1", pad='SAME')
#         pyramid2 = self.sep_conv_mobilenet(input, 3, 24, 1, "p2", pad='SAME')
#         pyramid3 = self.sep_conv_mobilenet(input, 3, 24, 1, "p3", pad='SAME')
#         pyramid4 = self.conv2d(input, 24, 1, 1, 'p4')

#         spp_concat = tf.concat([pyramid1, pyramid2, pyramid3, pyramid4], axis=3, name="sppConcat")

#         return spp_concat

#     def call(self, inputs):
#         conv1 = self.conv2d(inputs, 6, 3, 2, 'conv1')

#         conv2 = self.sep_conv_mobilenet(conv1, 3, 12, 1, "conv2")
#         conv2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(conv2)

#         conv3 = self.sep_conv_mobilenet(conv2, 3, 12, 1, "conv3")
#         conv3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(conv3)

#         spp = self.spp(conv3)

#         pooling = self.conv2d(conv1, 12, 1, 1, 'pooling')

#         spp_merg = self.conv2d(spp, 48, 1, 1, 'spp-merg')
#         o1 = tf.image.resize(spp_merg, [120, 160], method=tf.image.ResizeMethod.BILINEAR)

#         concat = tf.concat([o1, pooling], axis=3, name="concat")

#         o2 = self.sep_conv_mobilenet(concat, 3, NUMBER_OF_CLASSES, 1, "o2", 1)

#         out = tf.image.resize(o2, [OUTPUT_HEIGHT, OUTPUT_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

#         # Softmax activation at the end
#         out = tf.keras.layers.Softmax()(out)
        
#         return out





class MazeNet(tf.keras.Model):
    def __init__(self):
        super(MazeNet, self).__init__()
        
        self.conv1 = layers.Conv2D(6, kernel_size=3, strides=2, padding='same', activation='relu', name='conv1')
        
        self.conv2_sep = layers.SeparableConv2D(12, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2')
        self.avg_pool1 = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')
        
        self.conv3_sep = layers.SeparableConv2D(12, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3')
        self.avg_pool2 = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')
        
        self.spp_p1 = layers.SeparableConv2D(24, kernel_size=3, strides=1, padding='same', activation='relu', name='p1')
        self.spp_p2 = layers.SeparableConv2D(24, kernel_size=3, strides=1, padding='same', activation='relu', name='p2')
        self.spp_p3 = layers.SeparableConv2D(24, kernel_size=3, strides=1, padding='same', activation='relu', name='p3')
        self.spp_p4 = layers.Conv2D(24, kernel_size=1, strides=1, padding='same', activation='relu', name='p4')
        
        self.spp_merg = layers.Conv2D(48, kernel_size=1, strides=1, padding='same', activation='relu', name='spp-merg')
        
        self.o2_sep = layers.SeparableConv2D(NUMBER_OF_CLASSES, kernel_size=3, strides=1, padding='same', activation='relu', name='o2')
        
    def call(self, inputs, training=False):

        conv1_output = self.conv1(inputs)
        
        conv2_output = self.conv2_sep(conv1_output)
        conv2_output = self.avg_pool1(conv2_output)
        
        conv3_output = self.conv3_sep(conv2_output)
        conv3_output = self.avg_pool2(conv3_output)
        
        spp_p1 = self.spp_p1(conv3_output)
        spp_p2 = self.spp_p2(conv3_output)
        spp_p3 = self.spp_p3(conv3_output)
        spp_p4 = self.spp_p4(conv3_output)
        
        spp_concat = layers.Concatenate(name="sppConcat")([spp_p1, spp_p2, spp_p3, spp_p4])
        
        spp_merg_output = self.spp_merg(spp_concat)
        spp_merged_output_resized = tf.image.resize(spp_merg_output, [120, 160])
        
        pooling_output = layers.Conv2D(12, kernel_size=1, strides=1, padding='same', activation='relu', name='pooling')(conv1_output)
        
        concat_output = layers.Concatenate(name="concat")([spp_merged_output_resized, pooling_output])
        
        o2_output = self.o2_sep(concat_output)
        
        out = tf.image.resize(o2_output, [OUTPUT_HEIGHT, OUTPUT_WIDTH])

        out = layers.Softmax()(out)
        
        return out