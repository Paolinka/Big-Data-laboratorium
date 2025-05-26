from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Zadanie 2
fashion_mnist = fashion_mnist.load_data()

# Załadowanie danych z zestawu Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist

# Zadanie 3
# Normalizacja wartości pikseli do zakresu [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Konwersja etykiet do formatu one-hot encoding
enc = OneHotEncoder(sparse_output=False)
y_train = enc.fit_transform(y_train.reshape(-1, 1))
y_test = enc.transform(y_test.reshape(-1, 1))

# Zadanie 4
# Zmiana kształtu danych wejściowych do formatu (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Definicja modelu CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Zadanie 5
# Trenowanie modelu
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Zadanie 6
# Ocena modelu 
test_loss, test_acc = model.evaluate(x_test, y_test) 
print('Dokładność klasyfikacji:', test_acc)

# Zadanie 7
# Przeprowadzenie predykcji na danych testowych
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Obliczenie macierzy pomyłek
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Macierz pomyłek:")
print(conf_matrix)

# Wyświetlenie raportu klasyfikacji
class_report = classification_report(y_true_classes, y_pred_classes)
print("Raport klasyfikacji:")
print(class_report)


# Zadanie 8
# Wnioski:
'''
- Klasy z wysoką dokładnością wskazują, że model dobrze je rozpoznaje.
- Klasy z niską dokładnością mogą wymagać dodatkowej analizy, np. sprawdzenia podobieństwa do innych klas lub zwiększenia liczby próbek w danych treningowych.
- W macierzy pomyłek mozna zaobserwować, że najczęściej mylone są klasy 0 i 6, czyli T-shirt/top i shirt.
  Może to wynikać z tego, że są one podobne do siebie pod względem wyglądu.
- Mylone są też klasy 2 i 4, czyli Pullover i Coat, co również może być spowodowane podobieństwem tych ubrań.
'''