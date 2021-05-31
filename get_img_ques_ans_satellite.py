from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import numpy as np
import spacy
from spacy.lang.en import English
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from keras.utils import np_utils
import tensorflow as tf
import keras.backend as K
import gc
import re
import sys
np.set_printoptions(threshold=sys.maxsize)


# GLOABAL VARIABLES 
DATASET_LENGTH = 4511



def get_image_model():
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model



def get_vector_for_image_sat(img_path):
    '''
        input: image path
        Return (1,4096) vector
    '''
    #TO DO 
    # Load VGG16 model 
    # open image file
    # Return Image vector
    # load an image from file
    image = load_img(img_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # get extracted features
    features = model(image)
    return features


def get_timeseries_vector_ques_sat(question_string):
    '''
        input: question String
        output: numpy array of length 300
    '''
    # TO DO
    # Load Spacy model
    # Calculate vector for each word
    # Calculate timeseries vector from each word 
    # return timeseries vector  
    nlp = spacy.load('en_core_web_md')
    question_string = re.sub('[^a-zA-Z0-9 \n\.]', '',question_string)
    tknz = English(remove_separators = True,remove_symbols = True)
    tokens = tknz(question_string)
    tokens = [token.text for token in tokens]
    print(tokens)
    word2vec_dimension = 300
    len_of_timeseries = 15 # maximum length of question - arbirtarly chosen number since the maximum length was 25
    len_of_ques_tkns = len(tokens)
    question_tensor = np.zeros((len_of_timeseries,word2vec_dimension))
    for i in range(len_of_ques_tkns):
        qtens = nlp(tokens[i]).vector
        #print(qtens)
        question_tensor[i] = qtens
        #print(question_tensor[i])
    return question_tensor