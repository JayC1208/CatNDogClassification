# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
import tensorflow as tf
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def test():
	final_model = tf.keras.models.load_model('./model_saved')

	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	test_it = datagen.flow_from_directory('./test/', class_mode='binary', batch_size=64, target_size=(224, 224))
	# evaluate model
	_, acc = final_model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))

test()