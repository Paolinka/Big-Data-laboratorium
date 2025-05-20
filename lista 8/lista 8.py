from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_olivetti_faces, load_diabetes
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



def kodowanie_zmiennych_kategorycznych(dane):
    """
    Funkcja koduje zmienne kategoryczne w ramce danych
    Parametry:
    - dane: ramka danych
    Zwraca:
    - przetworzone dane
    """
    dane_imputed = dane.copy()

    # Kodowanie zmiennych kategorycznych
    label_encoder = LabelEncoder()
    categorical_columns = dane.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dane_imputed[col] = label_encoder.fit_transform(dane[col].astype(str))

    print("\nPrzetworzone dane:")
    print(dane_imputed.head())

    return dane_imputed


def K_Means_clustering(dane, feature_1: str, feature_2: str, n_clusters):
    """
    Funkcja wykonuje klasteryzację KMeans
    Parametry:
    - dane: ramka danych
    - feature_1: nazwa pierwszej cechy
    - feature_2: nazwa drugiej cechy
    - n_clusters: liczba klastrów
    Zwraca:
    - None
    """

    n = len(n_clusters) if isinstance(n_clusters, list) else 1
    n_rows = n//2 if n % 2 == 0 else (n//2) + 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, k in enumerate(n_clusters if isinstance(n_clusters, list) else [n_clusters]):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dane) 
    
        # Przewidywanie przynależności do klastrów dla każdej próbki 
        labels = kmeans.labels_ 
    
        # Wyświetlenie wyników 
        axes[i].scatter(dane[feature_1], dane[feature_2], c=labels, cmap='viridis', alpha=0.5) 
        axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red') 
        axes[i].set_xlabel(feature_1) 
        axes[i].set_ylabel(feature_2) 
        axes[i].set_title(f'K = {k}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.suptitle(f'Grupowanie za pomocą K-Means')
    plt.show() 
    
    
def Mean_Shift_clustering(dane, feature_1: str, feature_2: str):
    
    # Implementacja metody Mean Shift 
    ms = MeanShift(max_iter=1000)
    print('Mean Shift - work in progress...')
    ms.fit(dane) 
    cluster_centers = ms.cluster_centers_ 
    
    # Zwizualizowanie wyników 
    plt.figure(figsize=(10, 7)) 
    plt.scatter(dane[feature_1], dane[feature_2], c=ms.labels_, cmap='viridis') 
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=300, 
                edgecolor='k', facecolor='none') 
    plt.title('Metoda przesunięcia średniej (Mean Shift)') 
    plt.xlabel(feature_1) 
    plt.ylabel(feature_2) 
    plt.show()


def Agglomerative_clustering(dane, feature_1: str, feature_2: str, n_clusters):

    n = len(n_clusters) if isinstance(n_clusters, list) else 1
    n_rows = n//2 if n % 2 == 0 else (n//2) + 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, k in enumerate(n_clusters if isinstance(n_clusters, list) else [n_clusters]):
        # Implementacja klastrowania aglomeracyjnego 
        ac = AgglomerativeClustering(n_clusters=k) 
        ac.fit(dane) 
        # Przewidywanie przynależności do klastrów dla każdej próbki 
        labels = ac.labels_ 
    
        # Wyświetlenie wyników 
        axes[i].scatter(dane[feature_1], dane[feature_2], c=labels, cmap='viridis', alpha=0.5)  
        axes[i].set_xlabel(feature_1) 
        axes[i].set_ylabel(feature_2) 
        axes[i].set_title(f'K = {k}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Klastrowanie aglomeracyjne (Agglomerative Clustering)')
    plt.show()


def Gaussian_Mixture_clustering(dane, feature_1: str, feature_2: str, n_clusters):
    
    n = len(n_clusters) if isinstance(n_clusters, list) else 1
    n_rows = n//2 if n % 2 == 0 else (n//2) + 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, k in enumerate(n_clusters if isinstance(n_clusters, list) else [n_clusters]):
        # Implementacja klastrowania z użyciem mieszanki Gaussa 
        gmm = GaussianMixture(n_components=k) 
        gmm.fit(dane) 
        labels = gmm.predict(dane)
    
        # Wyświetlenie wyników 
        axes[i].scatter(dane[feature_1], dane[feature_2], c=labels, cmap='viridis', alpha=0.5)  
        axes[i].set_xlabel(feature_1) 
        axes[i].set_ylabel(feature_2) 
        axes[i].set_title(f'K = {k}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Klastrowanie z użyciem mieszanki Gaussa')
    plt.show()
 

def DBSCAN_clustering(dane):

    # Implementacja DBSCAN 
    dbscan = DBSCAN(eps=0.5, min_samples=5) 
    clusters = dbscan.fit_predict(dane) 
    
    print(f"ilość klastrów: {len(set(clusters))}")
    # Redukcja wymiarów do wizualizacji 
    pca = PCA(n_components=2).fit(dane) 
    X_pca = pca.transform(dane) 
    
    # Zwizualizowanie wyników 
    plt.figure(figsize=(10, 7)) 
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis') 
    plt.title('DBSCAN') 
    plt.xlabel('PCA 1') 
    plt.ylabel('PCA 2') 
    plt.colorbar(label='Klastry') 
    plt.show()


def hierarchical_clustering(dane):
    # Implementacja klastrowania hierarchicznego 
    linked = linkage(dane, method='ward') 
    
    # Wyświetlenie dendrogramu 
    plt.figure(figsize=(10, 7)) 
    dendrogram(linked, 
               orientation='top', 
               distance_sort='descending', 
               show_leaf_counts=True) 
    plt.title('Dendrogram Hierarchicznego Klastrowania') 
    plt.xlabel('Indeksy próbek') 
    plt.ylabel('Odległość') 
    plt.show()


def ocena_jakosci_klastrowania(dane, labels, nazwa_metody):
    sil_score = silhouette_score(dane, labels)
    ch_score = calinski_harabasz_score(dane, labels)
    db_score = davies_bouldin_score(dane, labels)
    print(f"\nMetoda: {nazwa_metody}")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.3f}")
    print(f"Davies-Bouldin Score: {db_score:.3f}")



# Zadanie 1
credit_card_data = pd.read_csv('lista 8\Credit_Card.csv')

# Usuwanie brakujących wartości
credit_card_data = credit_card_data.dropna()
credit_card_data = credit_card_data.drop_duplicates()

# Kodowanie zmiennych kategorycznych
credit_card_data = kodowanie_zmiennych_kategorycznych(credit_card_data)


# Zadanie 2
# Wybór cech do wykresu 2D
feature_1 = 'LIMIT_BAL'
feature_2 = 'BILL_AMT1'

# Przykładowe liczby klastrów do przetestowania
n_clusters = [2, 3, 4, 5]

K_Means_clustering(credit_card_data, feature_1, feature_2, n_clusters)

# Zadanie 3
# Standardyzacja danych
scaler = StandardScaler()
feature_columns = [feature_1, feature_2]
credit_card_data[feature_columns] = scaler.fit_transform(credit_card_data[feature_columns])

# Klasteryzacja z użyciem Mean Shift
Mean_Shift_clustering(credit_card_data, feature_1, feature_2)


# Zadanie 4
# Wczytanie zbioru danych Iris
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data = iris_data.dropna()
iris_data = iris_data.drop_duplicates()
iris_data = kodowanie_zmiennych_kategorycznych(iris_data)

# Wybór cech do wykresu 2D
feature_1 = 'sepal length (cm)'
feature_2 = 'petal length (cm)'

# Klasteryzacja aglomeracyjna
Agglomerative_clustering(iris_data, feature_1, feature_2, n_clusters=n_clusters)


# Zadanie 5
# Wczytanie zbioru danych Wine
wine = load_wine()
wine_data = pd.DataFrame(data=wine.data, columns=wine.feature_names)

wine_data = wine_data.dropna()
wine_data = wine_data.drop_duplicates()
wine_data = kodowanie_zmiennych_kategorycznych(wine_data)

# Wybór cech do wykresu 2D
feature_1 = 'malic_acid'
feature_2 = 'color_intensity'

Gaussian_Mixture_clustering(wine_data, feature_1, feature_2, n_clusters=n_clusters)


# Zadanie 6
# Wczytanie zbioru danych Olivetti Faces
faces = fetch_olivetti_faces()
faces_data = pd.DataFrame(data=faces.data)
faces_data = faces_data.dropna()
faces_data = faces_data.drop_duplicates()
faces_data = kodowanie_zmiennych_kategorycznych(faces_data)

DBSCAN_clustering(faces_data)

# Zadanie 7
diabetes = load_diabetes()
diabetes_data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_data = diabetes_data.dropna()
diabetes_data = diabetes_data.drop_duplicates()
diabetes_data = kodowanie_zmiennych_kategorycznych(diabetes_data)

hierarchical_clustering(diabetes_data)

# Zadanie 8
heart_data = pd.read_csv('lista 7\heart_failure_clinical_records_dataset.csv')
heart_data = heart_data.dropna()
heart_data = heart_data.drop_duplicates()
heart_data = kodowanie_zmiennych_kategorycznych(heart_data)

# Wybór cech do wykresu 2D
feature_1 = 'platelets'
feature_2 = 'creatinine_phosphokinase'

# Klasteryzacja KMeans
K_Means_clustering(heart_data, feature_1, feature_2, n_clusters=n_clusters)

# Klasteryzacja aglomeracyjna
Agglomerative_clustering(heart_data, feature_1, feature_2, n_clusters=n_clusters)

# Klasteryzacja z użyciem mieszanki Gaussa
Gaussian_Mixture_clustering(heart_data, feature_1, feature_2, n_clusters=n_clusters)

# Klasteryzacja DBSCAN
DBSCAN_clustering(heart_data)

# Przygotowanie danych do oceny (tylko wybrane cechy)
feature_columns = [feature_1, feature_2]
X = heart_data[feature_columns]

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
ocena_jakosci_klastrowania(X, kmeans_labels, "KMeans")

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(X)
ocena_jakosci_klastrowania(X, agg_labels, "Agglomerative Clustering")

# Gaussian Mixture
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit(X).predict(X)
ocena_jakosci_klastrowania(X, gmm_labels, "Gaussian Mixture")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
# Sprawdź, czy DBSCAN znalazł więcej niż 1 klaster (poza szumem)
if len(set(dbscan_labels)) > 1 and -1 in dbscan_labels:
    mask = dbscan_labels != -1
    unique_labels = set(dbscan_labels[mask])
    if len(unique_labels) > 1:
        ocena_jakosci_klastrowania(X[mask], dbscan_labels[mask], "DBSCAN (bez szumu)")
    else:
        print("\nDBSCAN: znaleziono tylko jeden klaster po odfiltrowaniu szumu - metryki nie mają sensu.")
elif len(set(dbscan_labels)) > 1:
    ocena_jakosci_klastrowania(X, dbscan_labels, "DBSCAN")
else:
    print("\nDBSCAN: znaleziono tylko jeden klaster lub same szumy - metryki nie mają sensu.")

print("""
Interpretacja:
- Wyższy Silhouette Score i Calinski-Harabasz Score oznaczają lepsze rozdzielenie i zwartość klastrów.
- Niższy Davies-Bouldin Score oznacza lepszą jakość klastrowania.
- Najlepsza metoda to ta, która osiąga najwyższy Silhouette i Calinski-Harabasz oraz najniższy Davies-Bouldin.
- W kontekście medycznym, metoda powinna tworzyć wyraźnie oddzielone grupy, które mogą odpowiadać różnym profilom klinicznym pacjentów.
      
Najlepsze wyniki uzyskano dla KMeans i Agglomerative Clustering, które wykazały najlepsze metryki jakości klastrowania.
""")