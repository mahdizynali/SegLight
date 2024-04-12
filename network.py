from config import *

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