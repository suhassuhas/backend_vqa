import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from easy_vqa import get_train_questions,get_test_questions,get_answers
from tensorflow.keras.preprocessing.text import Tokenizer

def load_and_proccess_image(image_path):
    # Load image, then scale and shift pixel values to [-0.5, 0.5]
    im = img_to_array(load_img(r'..\\vqa_react\\public\\simple_images\\'+image_path))
    return im / 255 - 0.5


def tokenizer_for_question():
    train_qs, train_answers, train_image_ids = get_train_questions()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_qs)
    return tokenizer

def image_input(path):
    img = load_and_proccess_image(path)
    img_e = np.expand_dims(img,0)
    return img_e

def question_input(question):
    train_qs, train_answers, train_image_ids = get_train_questions()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_qs)
    qs = tokenizer.texts_to_matrix([question])
    return qs

def predict_ans_simple(imgpath,question):
    ims = image_input(imgpath)
    ques = question_input(question)
    model = keras.models.load_model('model.h5')
    ans = model.predict([ims,ques])
    f_ans = print_ans(ans)
    return f_ans

def print_ans(ans):
    ans_dict = {0:"circle",1:"green",2:"red",3:"gray",4:"yes",5:"teal",6:"black",7:"rectangle",8:"yellow",9:"triangle",10:"brown",11:"blue",12:"no"}
    ans_list = list(ans[0])
    max_index = ans_list.index(max(ans_list))
    ans_string = ans_dict[max_index]
    return ans_string

