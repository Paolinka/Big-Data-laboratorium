from sklearn import datasets
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc


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


def przetwarzanie_danych(dane):
    """
    Funkcja przetwarza dane
    Parametry:
    - dane: ramka danych
    Zwraca:
    - przetworzone dane
    """
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


def specificity_score(conf_matrix):
    """ 
    Funkcja oblicza specyficzność na podstawie macierzy pomyłek
    Parametry:
    - conf_matrix: macierz pomyłek
    Zwraca:
    - specyficzność (dla każdego przypadku klasyfikacji)
    """
    specificity_per_class = []
    for i in range(conf_matrix.shape[0]):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)
    return specificity_per_class


def ocena_modelu(model, X_test, y_test):
    """
    Funkcja ocenia model
    Parametry:
    - model: model do oceny
    - X_test: dane testowe
    - y_test: etykiety testowe
    Zwraca:
    - dokładność modelu
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    confusion = confusion_matrix(y_test, predictions)
    specificity = specificity_score(confusion)
    
    return accuracy, precision, recall, confusion, specificity


def zbuduj_modele_i_wykonaj_ocene(dataframe, target = 'target', chosen_models=['KNN', 'SVM', 'Logistic Regression', 'Decision Tree'], roc_curve=False):
    """
    Funkcja buduje model i wykonuje ocenę
    Parametry:
    - dataframe: ramka danych
    - target: etykieta docelowa
    - chosen_models: lista wybranych modeli do oceny
    - roc_curve: czy rysować krzywą ROC
    Zwraca:
    - dokładność różnych modeli
    - ważność cech
    """
    # Przygotowanie danych do klasyfikacji
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja modeli
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='linear', probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Ocena modeli
    results = {}
    feature_importances = {}
    
    if len(chosen_models)%2 == 1:
        n_rows = len(chosen_models)//2 + 1
    else:
        n_rows = len(chosen_models)//2

    # Tworzenie wykresów
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 8))

    axes = axes.flatten()
    for i, model_name in enumerate(chosen_models):
        model = models[model_name]
        model.fit(X_train, y_train)
        accuracy, precision, recall, confusion, specificity = ocena_modelu(model, X_test, y_test)
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion,
            'specificity': specificity
        }

        print(f"\nOcena modelu {model_name}:")
        print(f"Dokładność: {accuracy:.2f}")
        print(f"Czułość: {recall:.2f}")
        print("Macierz pomyłek:")
        print(confusion)
        print("Specyficzność dla każdej klasy:", [float(s) for s in specificity])

        # Ważność cech
        if hasattr(model, 'feature_importances_'):
            feature_importances[model_name] = dict(zip(X.columns, model.feature_importances_))
        else:
            feature_importances[model_name] = None

        # Wykresy macierzy pomyłek
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(f"Macierz pomyłek {model_name}")
        axes[i].set_xlabel("Etykieta przewidywana")
        axes[i].set_ylabel("Etykieta rzeczywista")
        axes[i].set_xticks(ticks=np.arange(len(dataframe[target].unique())) + 0.5, labels=dataframe[target].unique())
        axes[i].set_yticks(ticks=np.arange(len(dataframe[target].unique())) + 0.5, labels=dataframe[target].unique(), rotation=0)
        

    plt.tight_layout()
    plt.show()
    
    if roc_curve:
        # Rysowanie krzywej ROC
        for model_name in chosen_models:
            model = models[model_name]
            y_scores = model.predict_proba(X_test)[:, 1]
            krzywa_roc(y_test, y_scores, model_name)
    

    return results, feature_importances
   

def krzywa_roc(y_test, y_scores, model_name):
    """
    Funkcja rysuje krzywą ROC
    Parametry:
    - y_test: etykiety testowe
    - y_scores: przewidywane prawdopodobieństwa
    - model_name: nazwa modelu
    Zwraca:
    - None
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle('Receiver Operating Characteristic')
    plt.title(f'Model: {model_name}')
    plt.legend(loc="lower right")
    plt.show()


# Zadanie 2
iris = datasets.load_iris()
zbadaj_strukture(iris)

# Konwersja do DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Przygotowanie danych do klasyfikacji
# Usunięcie brakujących wartości 
iris_df.dropna(inplace=True) 

# Usunięcie duplikatów 
iris_df.drop_duplicates(inplace=True)

# Przetwarzanie danych
iris_df = przetwarzanie_danych(iris_df)

# Trenowanie i ocena modeli
iris_results, _ = zbuduj_modele_i_wykonaj_ocene(iris_df)

# Wnioski:
# Wszystkie modele osiągnęły idealną dokładność 1.00, co sugeruje, że doskonale radzą sobie z danymi.
# W macierzy pomyłek nie wystąpiły żadne błędne klasyfikacje.



# Zadanie 3
breast_cancer = datasets.load_breast_cancer()
zbadaj_strukture(breast_cancer)

# Konwersja do DataFrame    
breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target

# Przygotowanie danych do klasyfikacji
# Usunięcie brakujących wartości
breast_cancer_df.dropna(inplace=True)

# Usunięcie duplikatów
breast_cancer_df.drop_duplicates(inplace=True)

# Przetwarzanie danych
breast_cancer_df = przetwarzanie_danych(breast_cancer_df)

# Trenowanie i ocena modeli
breast_cancer_results, breast_cancer_features = zbuduj_modele_i_wykonaj_ocene(breast_cancer_df)

# Wnioski:
# Wszystkie modele osiągnęły bardzo wysoką dokładność (w okolicach 0.95), co sugeruje, że dobrze radzą sobie z danymi.
# W przypadku klasyfikacji raka piersi groźniejsze są błędne klasyfikacje typu 1 (False Negative), ponieważ mogą prowadzić do nieodpowiedniego leczenia pacjentów.


# Zadanie 4
digits = datasets.load_digits()
zbadaj_strukture(digits)

# Spłaszczenie obrazów
n_samples = len(digits.images)
digits_flattened = digits.images.reshape((n_samples, -1))

# Konwersja do DataFrame
digits_df = pd.DataFrame(data=digits_flattened)
digits_df['target'] = digits.target

# Standaryzacja wartości pikseli
scaler = StandardScaler()
feature_columns = digits_df.drop('target', axis=1).columns
digits_df[feature_columns] = scaler.fit_transform(digits_df[feature_columns])

# Przygotowanie danych do klasyfikacji
# Usunięcie brakujących wartości
digits_df.dropna(inplace=True)

# Usunięcie duplikatów
digits_df.drop_duplicates(inplace=True)

# Przetwarzanie danych
digits_df = przetwarzanie_danych(digits_df)

# Trenowanie i ocena modeli
digits_results, _ = zbuduj_modele_i_wykonaj_ocene(digits_df, chosen_models=['SVM'])

# Analiza wyników dla zbioru digits
print("\nAnaliza wyników dla zbioru digits:")

# Pobranie wyników dla modelu SVM
svm_results = digits_results['SVM']
confusion = svm_results['confusion_matrix']

# Analiza błędów klasyfikacji
misclassification_counts = np.sum(confusion, axis=1) - np.diag(confusion)

# Wyświetlenie liczby błędnych klasyfikacji dla każdej cyfry
print("\nLiczba błędnych klasyfikacji dla każdej cyfry:")
for digit, count in enumerate(misclassification_counts):
    print(f"Cyfra {digit}: {count} błędnych klasyfikacji")

# Wnioski:
# Model SVM osiągnął bardzo wysoką dokładność, co sugeruje, że bardzo dobrze radzi sobie z danymi.
# W przypadku zbioru digits, najczęściej błędnie klasyfikowane cyfry to 9, 8 i 5.


# Zadanie 5
titanic = pd.read_csv('lista 7\Titanic-Dataset.csv')
print(titanic.head())
print(titanic.info())
print(titanic.describe())

# Przygotowanie danych do klasyfikacji
titanic = przetwarzanie_danych(titanic)
titanic.dropna(inplace=True)  # Usunięcie brakujących wartości
titanic.drop_duplicates(inplace=True)  # Usunięcie duplikatów
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)  # Usunięcie niepotrzebnych kolumn

# Standaryzacja wartości
scaler = StandardScaler()
feature_columns = titanic.drop('Survived', axis=1).columns
titanic[feature_columns] = scaler.fit_transform(titanic[feature_columns])

results_titanic, features_titanic = zbuduj_modele_i_wykonaj_ocene(titanic, target='Survived', roc_curve=True)

# Wyświetlenie najważniejszych cech dla każdego modelu
for model_name, feature_importance in features_titanic.items():
    if feature_importance is not None:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nNajważniejsze cechy dla modelu {model_name}:")
        for feature, importance in sorted_features[:5]:  # Top 5 cech
            print(f"{feature}: {importance:.4f}")
    else:
        print(f"\nModel {model_name} nie ma dostępnej ważności cech.")


# Wnioski:
# Wszystkie modele osiągnęły całkiem wysoką dokładność (w okolicach 0.8), ale KNN i Logistic Regression były najlepsze.
# W przypadku zbioru Titanic, najważniejsze cechy to sex, age, fare i class.

# Zadanie 6
heart_disease = pd.read_csv('lista 7\heart_failure_clinical_records_dataset.csv')
print(heart_disease.head())
print(heart_disease.info())
print(heart_disease.describe())

# Przygotowanie danych do klasyfikacji
heart_disease = przetwarzanie_danych(heart_disease)
heart_disease.dropna(inplace=True)  # Usunięcie brakujących wartości
heart_disease.drop_duplicates(inplace=True)  # Usunięcie duplikatów
heart_disease.drop(['time'], axis=1, inplace=True)  # Usunięcie niepotrzebnych kolumn

# Standaryzacja wartości
scaler = StandardScaler()
feature_columns = heart_disease.drop('DEATH_EVENT', axis=1).columns
heart_disease[feature_columns] = scaler.fit_transform(heart_disease[feature_columns])

# Trenowanie i ocena modeli
results_heart_disease, features_heart_disease = zbuduj_modele_i_wykonaj_ocene(heart_disease, target='DEATH_EVENT')

# Wyświetlenie najważniejszych cech dla każdego modelu
for model_name, feature_importance in features_heart_disease.items():
    if feature_importance is not None:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nNajważniejsze cechy dla modelu {model_name}:")
        for feature, importance in sorted_features[:5]:  # Top 5 cech
            print(f"{feature}: {importance:.4f}")
    else:
        print(f"\nModel {model_name} nie ma dostępnej ważności cech.")

# Wnioski:
# Wszystkie modele osiągnęły przeciętną dokładność, ale KNN i Logistic Regression były najlepsze.
# W przypadku zbioru heart_disease, najważniejsze cechy to age, ejection_fraction, serum_creatinine.
# Błędy w klasyfikacji mogą być spowodowane faktem, że target to śmierć pacjenta, a nie choroba serca.
