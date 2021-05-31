from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from get_img_ques_ans_vectors import get_vector_for_image,get_timeseries_vector_ques
import numpy as np
from tensorflow import keras 


#Load Model
model = keras.models.load_model("my_model_50K_almost_40_acc.h5")

# #Load Image 
# img_path = r"Z:\Desktop_SC\people.jpg"
# img_vec = get_vector_for_image(img_path)



# #Load Question 
# ques = "what are they playing?"
# ques_vec = get_timeseries_vector_ques(ques)

def predict_ans_complex(ques,img):
    img_vec = get_vector_for_image(img)
    ques_vec = get_timeseries_vector_ques(ques)
    #print(ques_vec.shape)
    #print(img_vec.shape)
    if (img_vec.ndim == 2):
        img_vec = np.array([img_vec])
    if (ques_vec.ndim == 2):
        ques_vec = np.array([ques_vec])
    #Predict Answer
    ans = model([img_vec,ques_vec])
    index = np.argmax(ans)
    ans_arr = np.load(r"answers.npy")
    print(ans_arr[index])
    return (ans_arr[index])





