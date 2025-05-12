import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, fetch_lfw_people, fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.cm as cm


def preprocess_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
    stop_words = set(stopwords.words('english')) # Tokenize and remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words] # Join tokens back to a single string
    return ' '.join(tokens)


# Zadanie 2
# Wczytaj dane
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redukcja wymiarowości za pomocą PCA
pca = PCA(n_components=2)  # Redukcja do 2 wymiarów
X_pca = pca.fit_transform(X_scaled)

# Wizualizacja wyników
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.7, lw=2, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Zadanie 3
data = load_digits()
X = data.data
y = data.target
target_names = data.target_names

# Redukcja wymiarowości przy użyciu t-SNE
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)

# Wizualizacja wyników
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], alpha=0.7, lw=2, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Wizualizacja danych przy użyciu t-SNE')
plt.xlabel('Wymiar 1')
plt.ylabel('Wymiar 2')
plt.show()

# Zadanie 4
# Wczytaj zbiór danych LFW
print("Startuję pobieranie zbioru danych LFW...")
data = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
print("Dane pobrane!")

X = data.data
n_samples, h, w = data.images.shape

print(f'Wczytano {n_samples} obrazów o wymiarach {h}x{w}')

# Wybierz liczbę komponentów
n_components = 8

# Tworzenie i dopasowanie modelu NMF
nmf_model = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=500)
W = nmf_model.fit_transform(X)
H = nmf_model.components_

print(f"Kształt macierzy cech (H): {H.shape}")

# Wyświetlenie cech jako obrazów
fig, axes = plt.subplots(2, 4, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(H[i].reshape(h, w))
    ax.set_title(f'Cecha {i+1}')
plt.suptitle('Podstawowe cechy wykryte przez NMF', fontsize=16)
plt.tight_layout()
plt.show()

# Wyświetlenie macierzy bazowych i wagowych
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(W, cmap='viridis', aspect='auto')
plt.title('Macierz bazowa (W)')
plt.xlabel('Składowe')
plt.ylabel('Próbki')
plt.subplot(1, 2, 2)
plt.imshow(H, cmap='viridis', aspect='auto')
plt.title('Macierz wagowa (H)')
plt.xlabel('Cechy')
plt.ylabel('Składowe')
plt.tight_layout()
plt.show()

# Zadanie 5
# Wczytanie danych
data = load_wine()
X = data.data  # Cechy
y = data.target  # Etykiety (klasy)

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Zastosowanie SVD
svd = TruncatedSVD(n_components=X.shape[1])  # Użyjemy maksymalnej liczby komponentów
X_svd = svd.fit_transform(X_scaled)

# Wariancja wyjaśniana przez poszczególne komponenty
explained_variance = svd.explained_variance_ratio_

# Wizualizacja wariancji wyjaśnianej przez każdy komponent
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Komponent')
plt.ylabel('Wariancja wyjaśniana')
plt.title('Wariancja wyjaśniana przez komponenty w SVD')
plt.show()

# Skumulowana wariancja
cumulative_variance = np.cumsum(explained_variance)

# Wyświetlenie skumulowanej wariancji
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='b')
plt.xlabel('Liczba komponentów')
plt.ylabel('Skumulowana wariancja')
plt.title('Skumulowana wariancja wyjaśniana przez komponenty w SVD')
plt.axhline(y=0.95, color='r', linestyle='--', label="95% wariancji")
plt.legend()
plt.show()

# Określenie minimalnej liczby komponentów wyjaśniających co najmniej 95% wariancji
k = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Optymalna liczba komponentów: {k}')

# Zredukowane dane
# Dekompozycja SVD
U, s, VT = np.linalg.svd(X_scaled)
reduced_data = np.dot(U[:, :k], np.diag(s[:k]))
print("Zredukowane dane:")
print(reduced_data)

# Wizualizacja danych w nowej przestrzeni o 2 komponentach z legendą
plt.figure(figsize=(8, 6))

colors = ['r', 'g', 'b'] 

for i, label in enumerate(np.unique(y)):
    plt.scatter(reduced_data[y == label, 0], reduced_data[y == label, 1], 
                label=f'Wino {label+1}', color=colors[i], edgecolor='k', s=100)
plt.legend(title='Odmiana wina')
plt.xlabel('Pierwszy komponent')
plt.ylabel('Drugi komponent')
plt.title('Redukcja wymiarowości danych Wine za pomocą SVD')
plt.show()


# Zadanie 6
# Pobranie zbioru danych 20 Newsgroups
newsgroups = fetch_20newsgroups(subset='all')

# Stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Przetwarzanie tekstu
newsgroups_data = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
newsgroups_data['text'] = newsgroups_data['text'].apply(preprocess_text)

# Przetwarzanie tekstu - tokenizacja, usuwanie stop words oraz wektoryzacja za pomocą Tfidf
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)

# Dopasowanie modelu i transformacja danych na macierz cech TF-IDF
X = vectorizer.fit_transform(newsgroups_data['text'])

# Wykorzystanie LDA do redukcji wymiarowości
lda = LDA(n_components=2)  # Redukujemy do dwóch wymiarów
X_lda = lda.fit_transform(X.toarray(), newsgroups_data['target']) 

cmap = cm.get_cmap('hsv', 20)

# Wizualizacja wyników
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=newsgroups_data['target'], cmap=cmap, alpha=0.5)
plt.colorbar(label='Klasa')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('Redukcja wymiarowości danych za pomocą LDA')
plt.show()
