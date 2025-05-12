import pandas as pd
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.cm as cm



def zbadaj_strukture(dane):
    """
    Funkcja bada strukturę ramki danych
    Parametry:
    - ramka danych
    Zwraca:
    - None
    """

    print("Pierwsze kilka wierszy danych:")
    print(dane.head())
    print("\nStruktura danych:")
    print(dane.info())


def obsluga_brakujacych_wartosci(dane):
    dane_imputed = dane.copy()

    # Imputacja danych liczbowych
    numeric_columns = dane.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    for col in numeric_columns:
        dane_imputed[[col]] = imputer.fit_transform(dane[[col]])

    # Kodowanie zmiennych kategorycznych
    label_encoder = LabelEncoder()
    categorical_columns = dane.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dane_imputed[col] = label_encoder.fit_transform(dane[col].astype(str))

    print("\nPrzetworzone dane:")
    print(dane_imputed.head())

    return dane_imputed


def wykonaj_redukcje(dane, n_components=2):
    # PCA (całość)
    pca_result = PCA(n_components=n_components).fit_transform(dane)

    # NMF (na danych >= 0)
    dane_nmf = np.abs(dane)  # NMF wymaga nieujemnych danych
    nmf_result = NMF(n_components=n_components, init='random', random_state=0, max_iter=500).fit_transform(dane_nmf)

    # t-SNE (na próbce)
    sampled = dane.sample(n=2000, random_state=42)
    tsne_result = TSNE(n_components=n_components, perplexity=30, n_iter=500, random_state=42).fit_transform(sampled)

    return pca_result, nmf_result, tsne_result


def pokaz_wykresy(pca, nmf, tsne):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(pca[:, 0], pca[:, 1], s=3, c='blue')
    axs[0].set_title("PCA")

    axs[1].scatter(nmf[:, 0], nmf[:, 1], s=3, c='green')
    axs[1].set_title("NMF")

    axs[2].scatter(tsne[:, 0], tsne[:, 1], s=3, c='red')
    axs[2].set_title("t-SNE (na próbce)")

    plt.suptitle("Porównanie technik redukcji wymiarowości", fontsize=14)
    plt.tight_layout()
    plt.show()


# Funkcja czyszcząca tekst
def clean_text(text):
    # Usuwanie znaków specjalnych, liczb, przekształcanie na małe litery
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text


def preprocess(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Użyj prostego podziału zamiast nltk.word_tokenize
    words = text.split()
    
    # Usuwanie stopwords i lematyzacja
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return ' '.join(words)





# # Zadanie 1
# # Wczytanie danych
# data = pd.read_csv('lista 6/rap_dataset.tsv', sep='\t')
# zbadaj_strukture(data)

# # Przetwarzanie
# data_clean = obsluga_brakujacych_wartosci(data)

# # Redukcja
# pca_result, nmf_result, tsne_result = wykonaj_redukcje(data_clean)

# # Wizualizacja
# pokaz_wykresy(pca_result, nmf_result, tsne_result)

# '''
# WNIOSKI
# PCA i NMF pokazują dane w formie bardziej rozciągniętej i globalnie uporządkowanej - dobre do eksploracji ogólnej struktury.
# t-SNE lepiej ujawnia lokalne grupowania i klastry - bardzo przydatne do analizy skupień.
# Różnice wynikają głównie z tego, czy technika skupia się na globalnych czy lokalnych zależnościach.
# '''

# # Zadanie 2
# data = pd.read_csv('lista 6\Twitter_Data.csv')
# zbadaj_strukture(data)

# # Obsługa brakujących wartości
# data['clean_text'] = data['clean_text'].fillna('')

# # Przekształcenie tekstu do cech numerycznych
# vectorizer = TfidfVectorizer(max_features=500)  # ograniczamy liczbę cech
# X = vectorizer.fit_transform(data['clean_text']).toarray()
# y = data['category']

# # Redukcja wymiarowości PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Wizualizacja
# colors = {1: 'green', 0: 'blue', -1: 'red'}
# labels = {1: 'Pozytywny', 0: 'Neutralny', -1: 'Negatywny'}

# plt.figure(figsize=(10, 6))
# for label in [-1, 0, 1]:
#     idx = y == label
#     plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=colors[label], label=labels[label], s=50, alpha=0.7)

# plt.title('Redukcja wymiarowości opinii z Twittera (PCA)')
# plt.xlabel('Składnik główny 1')
# plt.ylabel('Składnik główny 2')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # Zadanie 3
# data = pd.read_csv('lista 6\Twitter_Data.csv')

# # Obsługa brakujących wartości
# data = data.dropna(subset=['clean_text', 'category'])

# # TF-IDF
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X_tfidf = vectorizer.fit_transform(data['clean_text'])

# # Redukcja wymiarowości za pomocą SVD
# svd = TruncatedSVD(n_components=2, random_state=42)
# X_reduced = svd.fit_transform(X_tfidf)

# # Przygotowanie etykiet kolorów
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(data['category'])  # zakłada, że masz -1, 0, 1

# # Wykres
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='RdYlGn', s=10)
# plt.legend(handles=scatter.legend_elements()[0], labels=['negatywny', 'neutralny', 'pozytywny'])
# plt.title("Redukcja wymiarowości za pomocą SVD (TF-IDF)")
# plt.xlabel("SVD Component 1")
# plt.ylabel("SVD Component 2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # 3. Przykładowe dokumenty
# df = pd.DataFrame({
#     'original_text': data['clean_text'],
#     'label': data['category'],
#     'svd_dim1': X_reduced[:, 0],
#     'svd_dim2': X_reduced[:, 1]
# })

# # Sortujmy wg jednej ze składowych SVD
# df_sorted = df.sort_values('svd_dim1')

# # Wyświetl 5 najniższych i 5 najwyższych
# print("\nNajniższe wartości (svd_dim1):")
# print(df_sorted.head(5)[['label', 'svd_dim1', 'original_text']])

# print("\nNajwyższe wartości (svd_dim1):")
# print(df_sorted.tail(5)[['label', 'svd_dim1', 'original_text']])

# '''
# Najniższe wartości svd_dim1: Te zdania są krótkie, ogólne, neutralne lub niezwiązane bezpośrednio z tematyką polityczną. 
# Model SVD "uznał", że są mniej istotne w kontekście całej przestrzeni semantycznej.
# Najwyższe wartości svd_dim1: W tych tekstach: 1. Występuje nazwisko "Modi"
#                                               2. Mają emocjonalne nacechowanie (np. "dufferwal", "modi hater")
#                                               3. Odnoszą się do postaci politycznej
# '''

# Zadanie 4
data = pd.read_csv('lista 6\Articles.csv')

# Zastosuj funkcję czyszczącą do kolumny 'text'
data['cleaned_text'] = data['text'].apply(clean_text)

data['processed_text'] = data['cleaned_text'].apply(preprocess)

# Macierz dokument-słowo
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(data['processed_text'])

# Model LDA
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42) 
lda.fit(dtm)

# Wyciąganie słów kluczowych dla każdego tematu
def get_lda_topics(model, vectorizer, n_top_words=10):
    keywords = []
    for idx, topic in enumerate(model.components_):
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_top_words:]]
        keywords.append(words)
    return keywords

topics = get_lda_topics(lda, vectorizer)
for i, topic in enumerate(topics):
    print(f"Temat {i + 1}: {', '.join(topic)}")

# Przypisanie tematu do każdego artykułu
topic_assignments = lda.transform(dtm)
data['topic'] = topic_assignments.argmax(axis=1)

# Wizualizacja 2D
tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_values = tsne_model.fit_transform(topic_assignments)

# Wykres z kolorowaniem
cmap = cm.get_cmap('Set1', n_topics)  # ograniczenie liczby kolorów

plt.figure(figsize=(10, 6))

for topic in range(n_topics):
    indices = data['topic'] == topic
    plt.scatter(
        tsne_values[indices, 0],
        tsne_values[indices, 1],
        label=f'Temat {topic + 1}',
        alpha=0.7,
        c=[cmap(topic)]
    )

plt.title("Wizualizacja tematów LDA z t-SNE")
plt.legend(title="Tematy")
plt.show()


# Tabela krzyżowa: kategorie vs przypisane tematy
cross_tab = pd.crosstab(data['category'], data['topic'])

# Wyświetl tabelę
print("\nTabela krzyżowa (category vs. topic):\n")
print(cross_tab)

# Wizualizacja heatmapą
plt.figure(figsize=(12, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='coolwarm')
plt.title('Zależność między kategorią artykułu a przypisanym tematem LDA')
plt.ylabel('Kategoria artykułu')
plt.xlabel('Dominujący temat (LDA)')
plt.tight_layout()
plt.show()
