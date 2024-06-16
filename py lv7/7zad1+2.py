import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()


# definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])




# provedi treniranje mreze pomocu .fit()
history = model.fit(x_train_s, y_train_s, epochs=10, batch_size=128, validation_split=0.1)


# Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_accuracy = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test_s, y_test_s, verbose=0)

print(f'Točnost na skupu podataka za učenje: {train_accuracy:.4f}')
print(f'Točnost na skupu podataka za testiranje: {test_accuracy:.4f}')

# Prikazite matricu zabune na skupu podataka za testiranje
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()


# Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
incorrect_indices = np.where(y_pred_classes != y_true)[0]

plt.figure(figsize=(10, 4))
for i, incorrect in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[incorrect], cmap="gray")
    plt.title(f"Pred: {y_pred_classes[incorrect]}, Prava: {y_true[incorrect]}")
    plt.axis('off')
plt.show()

#matrica zabune pokazuje da je mal broj brojeva krivo klasificiran, te da je večina dobro sortirana
#po prikazu krivo sortiranih vidimo da je većinom klasifikator radio očekivane greške, npr. miješao okrugle brojeve, 6, 0 i 9
