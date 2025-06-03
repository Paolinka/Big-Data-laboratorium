from sklearn.datasets import load_breast_cancer, load_iris
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc




# Zadanie 2
breast_cancer_data = load_breast_cancer()

X = breast_cancer_data.data
y = breast_cancer_data.target

# Normalizacja wartości 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model MLP
model = Sequential([
    Dense(64, activation='relu', input_shape=(30,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Ocena modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Dokładność (accuracy):', test_acc)

# Predykcje na zbiorze testowym
y_pred = model.predict(X_test)
y_pred_classes = (y_pred >= 0.5).astype(int).ravel()
y_true_classes = y_test

# Obliczenie metryk
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes)
recall = recall_score(y_true_classes, y_pred_classes)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Specyficzność (specificity)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

print(f"Dokładność (accuracy): {accuracy:.4f}")
print(f"Precyzja (precision): {precision:.4f}")
print(f"Czułość (recall): {recall:.4f}")
print(f"Specyficzność (specificity): {specificity:.4f}")


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=breast_cancer_data.target_names,
            yticklabels=breast_cancer_data.target_names)
plt.title('Macierz pomyłek dla klasyfikacji raka piersi')
plt.xlabel('Etykiety przewidywane')
plt.ylabel('Etykiety rzeczywiste')
plt.tight_layout()
plt.show()

# zadanie 3
iris_data = load_iris()
X_iris = iris_data.data
y_iris = iris_data.target

# Normalizacja wartości
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)

# Konwersja etykiet do formatu one-hot encoding
y_iris = to_categorical(y_iris, num_classes=3)

# Podział danych na zbiór treningowy i testowy
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)


# Definicja modelu MLP
# Ma więcej warstw i neuronów, używa Dropout do zapobiegania przeuczeniu
model_iris = Sequential([
    Dense(128, activation='relu', input_shape=(4,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])


# Kompilacja modelu
model_iris.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model_iris.fit(X_train_iris, y_train_iris, epochs=30, validation_data=(X_test_iris, y_test_iris))

# Ocena modelu
test_loss_iris, test_acc_iris = model_iris.evaluate(X_test_iris, y_test_iris)
print('Dokładność (accuracy) dla zbioru Iris:', test_acc_iris)

# Predykcje na zbiorze testowym
y_pred_iris = model_iris.predict(X_test_iris)
y_pred_classes_iris = y_pred_iris.argmax(axis=1)
y_true_classes_iris = y_test_iris.argmax(axis=1)

# Obliczenie metryk
accuracy_iris = accuracy_score(y_true_classes_iris, y_pred_classes_iris)
precision_iris = precision_score(y_true_classes_iris, y_pred_classes_iris, average='weighted')
recall_iris = recall_score(y_true_classes_iris, y_pred_classes_iris, average='weighted')
conf_matrix_iris = confusion_matrix(y_true_classes_iris, y_pred_classes_iris)

# Specyficzność (specificity) dla każdej klasy osobno:
specificity_iris = []
for i in range(conf_matrix_iris.shape[0]):
    # True negatives for class i: sum of all elements except row i and column i
    tn = conf_matrix_iris.sum() - (conf_matrix_iris[i, :].sum() + conf_matrix_iris[:, i].sum() - conf_matrix_iris[i, i])
    # False positives for class i: sum of column i except diagonal
    fp = conf_matrix_iris[:, i].sum() - conf_matrix_iris[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_iris.append(specificity)

# Wyświetlenie wyników
print(f"Dokładność (accuracy) dla zbioru Iris: {accuracy_iris:.4f}")
print(f"Precyzja (precision) dla zbioru Iris: {precision_iris:.4f}")
print(f"Czułość (recall) dla zbioru Iris: {recall_iris:.4f}")
for idx, spec in enumerate(specificity_iris):
    print(f"Specyficzność (specificity) dla klasy {iris_data.target_names[idx]}: {spec:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_iris, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris_data.target_names,
            yticklabels=iris_data.target_names)
plt.title('Macierz pomyłek dla klasyfikacji zbioru Iris')
plt.xlabel('Etykiety przewidywane')
plt.ylabel('Etykiety rzeczywiste')
plt.tight_layout()
plt.show()


# Obliczenie krzywych ROC dla każdej klasy
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_iris[:, i], y_pred_iris[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Wyświetlenie krzywych ROC
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class {}) (area ={:.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# Zadanie 4
# Baza danych VGG Face nie jest ogólnodostępna

# Zadanie 5
# Być może to wina mojego laptopa, ale nie mogę pobrać COCO 2017 ze scikitlearn datasets


