import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

model = load_model('best_model.keras')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

def get_class_labels(train_dir):
    class_labels = sorted(os.listdir(train_dir))
    return class_labels

train_dir = 'G:\G Radna\PSU\py lv9\Train'

class_labels = get_class_labels(train_dir)

img_path = 'G:\G Radna\PSU\py lv9\znak.png' 

img_array = load_and_preprocess_image(img_path)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

print(f'Predicted class: {class_labels[predicted_class]}')

img = image.load_img(img_path)
plt.imshow(img)
plt.title(f'Predicted: {class_labels[predicted_class]}')
plt.axis('off')
plt.show()
