import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'G:\G Radna\PSU\py lv8\slika.png'

# Ucitaj sliku
img_original = mpimg.imread(filename)  
img = color.rgb2gray(img_original)
img = resize(img, (28, 28))

# Prikazi sliku
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')  
plt.show()

# Pripremi sliku - ulaz u mrezu
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32') / 255.0

# Ucitaj izgradenu mrezu
best_model = models.load_model('best_model.keras')

# Napravi predikciju za ucitanu sliku pomocu mreze
prediction = best_model.predict(img)
predicted_class = np.argmax(prediction)

# Ispis rezultat u terminal
print(f"Predicted class: {predicted_class}")

#klasifikacija lošije klasificira rotirane slike, ili slike na kojima broj nije na očekivanoj poziciji