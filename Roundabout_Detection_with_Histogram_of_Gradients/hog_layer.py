import tensorflow as tf
from keras.layers import Layer
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np

class HOGLayer(Layer):
    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, **kwargs):
        super(HOGLayer, self).__init__(**kwargs)
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def build(self, input_shape):
        # No trainable weights for this layer
        pass

    def call(self, inputs):
        def extract_hog(image):
            image = rgb2gray(image)
            hog_features, _ = hog(
                image,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                orientations=self.orientations,
                visualize=True,
                multichannel=False
            )
            return hog_features

        hog_features = tf.map_fn(lambda img: tf.numpy_function(extract_hog, [img], tf.float32), inputs, dtype=tf.float32)
        return hog_features

    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        hog_features_shape = (self.orientations * ((height // self.pixels_per_cell[0]) - 1) * ((width // self.pixels_per_cell[1]) - 1))
        return (input_shape[0], hog_features_shape)

    def get_config(self):
        config = super(HOGLayer, self).get_config()
        config.update({
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'orientations': self.orientations
        })
        return config
