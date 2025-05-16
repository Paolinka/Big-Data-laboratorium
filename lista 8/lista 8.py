from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def zbadaj_strukture(dane):
    """
    Funkcja bada strukturę ramki danych
    Parametry:
    - ramka danych
    Zwraca:
    - None
    """

    print("Pierwsze kilka wierszy danych:")
    print(dane.data[:5])
    print("\nPierwsze kilka etykiet:")
    print(dane.target[:5])
    print("\nStruktura danych:")
    print(dane.DESCR)


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

    # Wybór cech do grupowania 
    X = dane[[feature_1, feature_2]]

    n = len(n_clusters) if isinstance(n_clusters, list) else 1
    n_rows = n//2 if n % 2 == 0 else (n//2) + 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, k in enumerate(n_clusters if isinstance(n_clusters, list) else [n_clusters]):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X) 
    
        # Przewidywanie przynależności do klastrów dla każdej próbki 
        labels = kmeans.labels_ 
    
        # Wyświetlenie wyników 
        axes[i].scatter(X[feature_1], X[feature_2], c=labels, cmap='viridis', alpha=0.5) 
        axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red') 
        axes[i].set_xlabel(feature_1) 
        axes[i].set_ylabel(feature_2) 
        axes[i].set_title(f'K = {k}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.suptitle(f'Grupowanie za pomocą K-Means')
    plt.show() 
    
    
def Mean_Shift_clustering(dane, feature_1: str, feature_2: str):
    # Wybór cech do grupowania 
    X = dane[[feature_1, feature_2]]

    
    # Implementacja metody Mean Shift 
    ms = MeanShift(max_iter=1000)
    print('Mean Shift - work in progress...')
    ms.fit(X) 
    cluster_centers = ms.cluster_centers_ 
    
    # Zwizualizowanie wyników 
    plt.figure(figsize=(10, 7)) 
    plt.scatter(X[feature_1], X[feature_2], c=ms.labels_, cmap='viridis') 
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=300, 
                edgecolor='k', facecolor='none') 
    plt.title('Metoda przesunięcia średniej (Mean Shift)') 
    plt.xlabel(feature_1) 
    plt.ylabel(feature_2) 
    plt.show()


def Agglomerative_clustering(dane, feature_1: str, feature_2: str, n_clusters):

    # Wybór cech do grupowania 
    X = dane[[feature_1, feature_2]]

    n = len(n_clusters) if isinstance(n_clusters, list) else 1
    n_rows = n//2 if n % 2 == 0 else (n//2) + 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, k in enumerate(n_clusters if isinstance(n_clusters, list) else [n_clusters]):
        # Implementacja klastrowania aglomeracyjnego 
        ac = AgglomerativeClustering(n_clusters=k) 
        ac.fit(X) 
        # Przewidywanie przynależności do klastrów dla każdej próbki 
        labels = ac.labels_ 
    
        # Wyświetlenie wyników 
        axes[i].scatter(X[feature_1], X[feature_2], c=labels, cmap='viridis', alpha=0.5)  
        axes[i].set_xlabel(feature_1) 
        axes[i].set_ylabel(feature_2) 
        axes[i].set_title(f'K = {k}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Klastrowanie aglomeracyjne (Agglomerative Clustering)')
    plt.show()
 



# Zadanie 1
credit_card_data = pd.read_csv('lista 8\Credit_Card.csv')

# Usuwanie brakujących wartości
credit_card_data = credit_card_data.dropna()
credit_card_data = credit_card_data.drop_duplicates()

# Kodowanie zmiennych kategorycznych
credit_card_data = kodowanie_zmiennych_kategorycznych(credit_card_data)

# Zadanie 2
# Wybór cech do klasteryzacji
feature_1 = 'LIMIT_BAL'
feature_2 = 'BILL_AMT1'

# Przykładowe liczby klastrów do przetestowania
n_clusters = [2, 3, 4, 5]

# K_Means_clustering(credit_card_data, feature_1, feature_2, n_clusters)

# Zadanie 3
# Standardyzacja danych
scaler = StandardScaler()
feature_columns = [feature_1, feature_2]
credit_card_data[feature_columns] = scaler.fit_transform(credit_card_data[feature_columns])

# Klasteryzacja z użyciem Mean Shift
# Mean_Shift_clustering(credit_card_data, feature_1, feature_2)


# Zadanie 4
# Wczytanie zbioru danych Iris
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data = iris_data.dropna()
iris_data = iris_data.drop_duplicates()
iris_data = kodowanie_zmiennych_kategorycznych(iris_data)

# Wybór cech do klasteryzacji
feature_1 = 'sepal length (cm)'
feature_2 = 'sepal width (cm)'

# Klasteryzacja aglomeracyjna
Agglomerative_clustering(iris_data, feature_1, feature_2, n_clusters=n_clusters)




