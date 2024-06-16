from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# 1. Izgradnja konvolucijske neuronske mreže
model = models.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 2. Definiranje karakteristika procesa učenja
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 3. Definiranje callbackova
log_dir = "logs"
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 4. Treniranje mreže
history = model.fit(x_train_s, y_train_s,
                    batch_size=128,
                    epochs=10,
                    validation_split=0.1,
                    callbacks=[model_checkpoint_callback, tensorboard_callback])

# 5. Učitavanje najboljeg modela
best_model = models.load_model('best_model.keras')

# 6. Izračun točnosti na skupu podataka za učenje i testiranje
train_loss, train_accuracy = best_model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_accuracy = best_model.evaluate(x_test_s, y_test_s, verbose=0)

print(f'Točnost na skupu podataka za učenje: {train_accuracy:.4f}')
print(f'Točnost na skupu podataka za testiranje: {test_accuracy:.4f}')

# Prikaz matrice zabune na skupu podataka za testiranje
y_pred = best_model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Prikaz matrice zabune na skupu podataka za učenje
y_train_pred = best_model.predict(x_train_s)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_train_true = np.argmax(y_train_s, axis=1)

cm_train = confusion_matrix(y_train_true, y_train_pred_classes)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=range(10))
disp_train.plot(cmap=plt.cm.Blues)
plt.show()

#matrica zabune prikazuju visoku točnost rezultata