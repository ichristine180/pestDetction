
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from categories import categories
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array
load_model = tf.keras.models.load_model
model = load_model("model.h5")
#file_path = os.path.join('/home/chris/myapp/pestProject/Dataset/test/test.png')
def predictPest(img):
  test_image = load_img(img, target_size = (128, 128)) # load image 
  print("@@ Got Image for prediction")
#   test_image.show()
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image)
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result)
  print(pred)
  return pred
# res= predict(file_path)
# print("pest group ",categories()[int(res)])

