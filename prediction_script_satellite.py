from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from get_img_ques_ans_satellite import get_vector_for_image_sat,get_timeseries_vector_ques_sat
import numpy as np
from tensorflow import keras 

#Load Model
model = keras.models.load_model("satellite_model.h5")



def predict_ans_satellite(img_path,ques):
    #Load Image 
    img_vec = get_vector_for_image_sat(r"..\\vqa_react\\public\\satellite_images\\"+img_path)

    #Load Question 
    ques_vec = get_timeseries_vector_ques_sat(ques)

    #print(ques_vec.shape)
    #print(img_vec.shape)
    if (img_vec.ndim == 2):
        img_vec = np.array([img_vec])
    if (ques_vec.ndim == 2):
        ques_vec = np.array([ques_vec])

    #Predict Answer
    ans = model([img_vec,ques_vec])
    index = np.argmax(ans)
    ans_arr = np.load("answers_satellite.npy")
    print(ans_arr[index])
    return ans_arr[index]