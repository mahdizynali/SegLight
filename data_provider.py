from config import *

# crop_percent = 0.6
# channels = [3, 3]

# palette = cfg.color_palette
# num_classes = len(palette)
# input_width = cfg.INPUT_WIDTH
# input_height = cfg.INPUT_HEIGHT


# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# def image_example(image_string, label_string):
#     image_shape = tf.image.decode_png(image_string).shape
#     feature = {
#         'height': _int64_feature(image_shape[0]),
#         'width': _int64_feature(image_shape[1]),
#         'depth': _int64_feature(image_shape[2]),
#         'image': _bytes_feature(image_string),
#         'label': _bytes_feature(label_string)
#     }
#     return tf.train.Example(features=tf.train.Features(feature=feature))


# def _parse_image_function(example_proto):
#     """
#     Create a dictionary describing the features.
#     """
#     image_feature_description = {
#         'height': tf.io.FixedLenFeature([], tf.int64),
#         'width': tf.io.FixedLenFeature([], tf.int64),
#         'depth': tf.io.FixedLenFeature([], tf.int64),
#         'label': tf.io.FixedLenFeature([], tf.string),
#         'image': tf.io.FixedLenFeature([], tf.string)
#     }
#     exm = tf.io.parse_single_example(example_proto, image_feature_description)
    
#     image = tf.image.decode_png(exm['image'])
#     label = tf.image.decode_png(exm['label'])

#     # height = tf.cast(exm['height'], tf.int32)
#     # width = tf.cast(exm['width'], tf.int32)
#     image = tf.cast(image, dtype=tf.float32)
#     label = tf.cast(label, dtype=tf.float32)

#     # convert image and label from rgb to bgr
#     image = tf.reverse(image, axis=[2])
#     label = tf.reverse(label, axis=[2])
#     return image, label


# def _color(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

#     # x = tf.image.random_hue(x, 0.5)
#     x = tf.image.random_saturation(x, 0.8, 1.2)
#     x = tf.image.random_brightness(x, 0.05)
#     x = tf.image.random_contrast(x, 0.7, 1.7)
#     x = tf.clip_by_value(x, 0, 255)
#     # x = tf.image.per_image_standardization(x)
#     return x, y


# def _one_hot_encode(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
#     """
#     Converts mask to a one-hot encoding specified by the semantic map.
#     """
#     one_hot_map = []
#     for class_name in palette:
#         eq_list = []
#         for color in palette[class_name]:
#             eq = tf.equal(y, color)
#             rd = tf.reduce_all(eq, axis=-1)
#             eq_list.append(rd)
#         orl = tf.reduce_any(eq_list, axis=0)
#         # eq = tf.equal(y, palette[class_name])
#         class_map = tf.cast(orl, tf.float32)
#         one_hot_map.append(class_map)
#     one_hot_map = tf.stack(one_hot_map, axis=-1)
#     y = tf.cast(one_hot_map, tf.float32)
#     return x, y


# def _flip_left_right(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
#     """
#     Randomly flips image and mask left or right in accord.
#     """
#     x = tf.image.random_flip_left_right(x, seed=12)
#     y = tf.image.random_flip_left_right(y, seed=12)

#     return x, y


# def _crop_random(image, mask):
#     """
#     Randomly crops image and mask in accord.
#     """

#     cond_crop_image = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32, seed=2), tf.bool)
#     cond_crop_mask = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32, seed=2), tf.bool)


#     shape = tf.cast(tf.shape(image), tf.float32)
#     h = tf.cast(shape[0] * crop_percent, tf.int32)
#     w = tf.cast(shape[1] * crop_percent, tf.int32)

#     image = tf.cond(cond_crop_image, lambda: tf.image.random_crop(image, [h, w, channels[0]], seed=2), lambda: tf.identity(image))
#     mask = tf.cond(cond_crop_mask, lambda: tf.image.random_crop(mask, [h, w, channels[1]], seed=2), lambda: tf.identity(mask))


#     return image, mask


# def _resize_data(image, mask):
#     """
#     Resizes images to specified size.
#     """
#     image = tf.expand_dims(image, axis=0)
#     mask = tf.expand_dims(mask, axis=0)

#     image = tf.image.resize(image, (240, 320))

#     mask = tf.image.resize(mask, (240, 320), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#     image = tf.squeeze(image, axis=0)
#     mask = tf.squeeze(mask, axis=0)


#     return image, mask


# def one_hot_image_matrix_to_label(data):
#     label = np.zeros((input_height, input_width, 3), np.uint8())
#     for i in range(input_height):
#         for j in range(input_width):
#             label_index = np.argmax(data[i][j])
#             for class_name, class_index in zip(palette, range(num_classes)):
#                 if (label_index == class_index):
#                     label[i][j] = palette[class_name][0]
#     return label

# def label_matrix_to_label(data):
#     label = np.zeros((input_height, input_width, 3), np.uint8())
#     for i in range(input_height):
#         for j in range(input_width):
#             label_index = data[i][j]
#             for class_name, class_index in zip(palette, range(num_classes)):
#                 if (label_index == class_index):
#                     label[i][j] = palette[class_name][0]
#     return label

# def tfrecord_data_image_to_opencv_mat(image):
#     image = image.astype('uint8')
#     # frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
#     return image


# def cv_show_image(frame, frame_title, wait_time_ms):
#     cv2.imshow(frame_title, frame)
#     cv2.waitKey(wait_time_ms)





tf.executing_eagerly()

showSample = False
tfRAddress = './dataset'


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label_string):
    image_shape = tf.image.decode_png(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image': _bytes_feature(image_string),
        'label': _bytes_feature(label_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

writerTrain = tf.io.TFRecordWriter(tfRAddress + 'train.tfrecords')
writerTest = tf.io.TFRecordWriter(tfRAddress + 'test.tfrecords')

for i, address in enumerate(glob.glob(IMAGES_PATH + '*.png'), start=1):
    if len(address.split('_')) == 1:
        image = cv2.imread(address)
        label = cv2.imread("dataset/labels/" + address.split('/')[-1].split('.')[0] + '.png')

        # cv2.imshow("img", image)
        # cv2.imshow("label", label)
        # if cv2.waitKey(0) == ord('q'):
        #     break

        image_string = cv2.imencode('.png', image)[1].tostring()
        label_string = cv2.imencode('.png', label)[1].tostring()
        tf_example = image_example(image_string, label_string)
        if i <= TEST_SIZE:
            writerTest.write(tf_example.SerializeToString())
        else:
            writerTrain.write(tf_example.SerializeToString())

writerTrain.close()
writerTest.close()
