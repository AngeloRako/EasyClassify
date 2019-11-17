# load and evaluate a saved model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import load_model
import numpy as np
from numpy import loadtxt
from keras.preprocessing import image
import argparse

#Obtain model from file
def load_Model_from_file(modelname):
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # load model
    classifier.load_weights(modelname)
    return classifier

if __name__ == '__main__':
    #Arguments
    parser = argparse.ArgumentParser(description="Classify image")
    parser.add_argument('-i', '--image', type=str, required=True, help='Image to classify')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model to use.')

    args = parser.parse_args()

    #Predictions
    classifier = load_model(args.model)
    test_image = image.load_img(args.image, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #training_set.class_indices
    if result[0][0] == 1:
        prediction = 'bus'
    else:
        prediction = 'ambulance'

    print('\n\nI think this is a... ' + prediction)
