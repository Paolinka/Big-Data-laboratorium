import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


# zadanie 2
def generate_data(n):
    # Definiowanie listy, która będzie przechowywać dane
    data = []

    # Generowanie N losowych wierszy danych
    for _ in range(n):
        area = random.randint(30, 100)
        rooms = random.randint(1, 5)
        floor = random.randint(1, 10)
        year_of_construction = random.randint(1950, 2022)
        price = random.randint(11_000, 14_000) * area
        data.append([area, rooms, floor, year_of_construction, price])

    # Tworzenie obiektu DataFrame z listy danych
    df = pd.DataFrame(data, columns=['area', 'rooms', 'floor', 'year_of_construction', 'price'])

    # Zapisanie danych do pliku CSV
    df.to_csv('appartments.csv', index=False)

    print(f"Plik 'appartments.csv' został wygenerowany z {n} wierszami danych.")

def ocen_predykcje(y_predykcja, y_testowe):
    r2 = r2_score(y_testowe, y_predykcja)

    jakosc_dopasowania = {
        (0, 0.5): "dopasowanie niezadowalające",
        (0.5, 0.6): "dopasowanie słabe",
        (0.6, 0.8): "dopasowanie zadowalające",
        (0.8, 0.9): "dopasowanie dobre",
        (0.9, 1): "dopasowanie bardzo dobre"
    }

    for (min_wartosc, max_wartosc), opis in jakosc_dopasowania.items():
        if min_wartosc <= r2 < max_wartosc:
            print(f"\nWartość współczynnika determinacji R2: {r2}. Ocena jakości dopasowania: {opis}\n")


# Wywołanie funkcji generate_data() z określoną ilością wierszy danych (np.100)
generate_data(1000)

# Wczytanie zbioru danych z pliku CSV
df = pd.read_csv('appartments.csv')

# Zbadanie struktury danych
print("Struktura danych:")
print(df.head())
print("\nTypy danych:")
print(df.dtypes)

# Identyfikacja brakujących wartości i obsługa brakujących danych
missing_values = df.isnull().sum()

# appartments.csv nie ma brakujących wartości, ale robimy i tak
print("\nBrakujące wartości:")
print(missing_values)

# Dla brakujących wartości stosujemy imputacje wartości średnich
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Podział zbioru danych na zbiór treningowy i testowy
X = df_filled.drop('price', axis=1)  # usunięcie kolumny z wartościami celu
y = df_filled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wyświetlenie podsumowania zbioru danych treningowych i testowych
print("\nRozmiar zbioru treningowego:", X_train.shape)
print("Rozmiar zbioru testowego:", X_test.shape)

# Tworzenie modelu
model = LinearRegression()

# Dopasowanie modelu do danych
model.fit(X_train, y_train)

# Przewidywanie wartości
y_pred = model.predict(X_test)
print("Przewidywana wartość dla X_test:", y_pred)

# Ocena predykcji
ocen_predykcje(y_pred, y_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue')  # Linia idealnych przewidywań
plt.xlabel("Rzeczywiste ceny")
plt.ylabel("Przewidywane ceny")
plt.title("Porównanie przewidywanych i rzeczywistych cen mieszkań")
plt.show()


# zadanie 4

# Wczytanie danych
dane_temp = pd.read_csv("temperature_and_energy_consumption.csv", sep=',')

# Zbadanie struktury danych
print("Struktura danych:")
print(dane_temp.head())
print("\nTypy danych:")
print(dane_temp.dtypes)

# Wyświetlenie brakujących wartości
print("\nBrakujące wartości:")
print(missing_values)

# Imputacja wartości średnich (jeśli są NaN)
dane_temp.fillna({"temperature": dane_temp["temperature"].mean(),
                  "energy_consumption": dane_temp["energy_consumption"].mean()}, inplace=True)

# Konwersja dat i wyciągnięcie numeru miesiąca
dane_temp["time_n"] = pd.to_datetime(dane_temp["time_n"])
dane_temp["month"] = dane_temp["time_n"].dt.month

# Podział na zbiór treningowy i testowy
X = dane_temp[['month']]
y = dane_temp['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie cech wielomianowych
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Trenowanie modelu
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Przewidywanie temperatur dla każdego miesiąca (1-12)
X_pred = np.linspace(1, 12, 100).reshape(-1, 1)
X_pred_poly = poly.transform(X_pred)
y_pred = model.predict(X_pred_poly)

# Przewidywanie na zbiorze testowym
y_test_pred = model.predict(X_test_poly)

# Ocena predykcji
ocen_predykcje(y_test_pred, y_test)

# Wizualizacja wyników
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="blue", label="Rzeczywiste dane (trening)")
plt.scatter(X_test, y_test, color="green", label="Rzeczywiste dane (test)")
plt.plot(X_pred, y_pred, color="red", label="Regresja wielomianowa")
plt.xlabel("Miesiąc")
plt.ylabel("Temperatura (°C)")
plt.title("Regresja wielomianowa: miesiąc vs temperatura")
plt.xticks(np.arange(1, 13, 1))
plt.legend()
plt.grid()
plt.show()



# zadanie 5
# Przygotowanie danych
X = dane_temp[['temperature']]
y = dane_temp['energy_consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie modelu
alphas = [0.1, 0.3, 0.5, 0.001]  # Parametr regularyzacji
modele_grzbietowe = [Ridge(alpha=alpha) for alpha in alphas]
modele_lasso = [Lasso(alpha=alpha) for alpha in alphas]

# Tworzenie siatki wykresów
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()  # Spłaszczenie dla łatwego iterowania

# Wykresy regresji grzbietowej
for i in range(len(alphas)):
    model = modele_grzbietowe[i]
    alpha = alphas[i]

    # Dopasowanie modelu do danych
    model.fit(X_train, y_train)

    # Przewidywanie wartości
    y_pred = model.predict(X_test)
    print(f"Regresja grzbietowa. Alpha = {alpha}. Przewidywana wartość dla X_test: {y_pred}")

    # Ocena predykcji
    ocen_predykcje(y_pred, y_test)

    # Przedstawienie wyników
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[i])
    axes[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linia idealnych przewidywań
    axes[i].set_xlabel("Rzeczywiste zużycie energii")
    axes[i].set_ylabel("Przewidywane zużycie energii")
    axes[i].set_title(f"Regresja grzbietowa. Alpha = {alpha}")

# Wykresy regresji Lasso
for i in range(len(alphas)):
    model = modele_lasso[i]
    alpha = alphas[i]

    # Dopasowanie modelu do danych
    model.fit(X_train, y_train)

    # Przewidywanie wartości
    y_pred = model.predict(X_test)
    print(f"Regresja Lasso. Alpha = {alpha}. Przewidywana wartość dla X_test: {y_pred}")

    # Ocena predykcji
    ocen_predykcje(y_pred, y_test)

    # Przedstawienie wyników
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[i+len(alphas)])
    axes[i+len(alphas)].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linia idealnych przewidywań
    axes[i+len(alphas)].set_xlabel("Rzeczywiste zużycie energii")
    axes[i+len(alphas)].set_ylabel("Przewidywane zużycie energii")
    axes[i+len(alphas)].set_title(f"Regresja Lasso. Alpha = {alpha}")

# Tworzenie modelu liniowego
model = LinearRegression()

# Dopasowanie modelu do danych
model.fit(X_train, y_train)

# Przewidywanie wartości
y_pred = model.predict(X_test)
print("Regresja liniowa. Przewidywana wartość dla X_test:", y_pred)

# Ocena predykcji
ocen_predykcje(y_pred, y_test)

# Dodanie wykresu
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[-1])
axes[-1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linia idealnych przewidywań
axes[-1].set_xlabel("Rzeczywiste zużycie energii")
axes[-1].set_ylabel("Przewidywane zużycie energii")
axes[-1].set_title("Regresja liniowa")

# Wyświetlenie wszystkich wykresów
plt.tight_layout()  # Lepszy układ
plt.show()


# zadanie 6
# Wczytanie danych
dane_med = pd.read_csv("dane_medyczne.csv", sep=',')

# Zbadanie struktury danych
print("Struktura danych:")
print(dane_med.head())
print("\nTypy danych:")
print(dane_med.dtypes)

# Identyfikacja brakujących wartości i obsługa brakujących danych
missing_values = dane_med.isnull().sum()

# appartments.csv nie ma brakujących wartości, ale robimy i tak
print("\nBrakujące wartości:")
print(missing_values)

# Dla brakujących wartości stosujemy imputacje wartości średnich
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(dane_med), columns=dane_med.columns)

# Podział zbioru danych na zbiór treningowy i testowy
X = df_filled.drop('czas_przezycia', axis=1)  # usunięcie kolumny z wartościami celu
y = df_filled['czas_przezycia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wyświetlenie podsumowania zbioru danych treningowych i testowych
print("\nRozmiar zbioru treningowego:", X_train.shape)
print("Rozmiar zbioru testowego:", X_test.shape)

# Tworzenie modelu
kernels = ['linear', 'poly', 'rbf']
reg_params = [0.1, 1, 10, 100]

# lista wartości R2 aby wybrać najlepszą konfigurację
r_squares = np.array([[0 for _ in range(len(reg_params))] for _ in range(len(kernels))], np.float64)

# Tworzenie siatki wykresów
fig, axes = plt.subplots(3, 4, figsize=(12, 8))

# Wykresy regresji SVR
for i in range(len(kernels)):
    for j in range(len(reg_params)):

        model = SVR(kernel=kernels[i], C=reg_params[j])

        # Dopasowanie modelu do danych
        model.fit(X_train, y_train)

        # Przewidywanie wartości
        y_pred = model.predict(X_test)
        print(f"Regresja SVR. Kernel: {kernels[i]}. C: {reg_params[j]}. Przewidywana wartość dla X_test: {y_pred}")

        # Ocena predykcji
        ocen_predykcje(y_pred, y_test)
        r_squares[i, j] = r2_score(y_test, y_pred)

        # Przedstawienie wyników
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[i, j])
        axes[i, j].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linia idealnych przewidywań
        axes[i, j].set_xlabel("Rzeczywisty czas przeżycia pacjenta")
        axes[i, j].set_ylabel("Predykcja")
        axes[i, j].set_title(f"Regresja SVR. Kernel: {kernels[i]}. C: {reg_params[j]}")

# Wyświetlenie wszystkich wykresów
plt.tight_layout()  # Lepszy układ
plt.show()


# Znalezienie indeksu dla największego R2 - najlepsza konfiguracja SVR
max_index = np.unravel_index(np.argmax(r_squares), r_squares.shape)

# Różne modele do porównania z najlepszym SVR
modele = [SVR(kernel=kernels[max_index[0]], C=reg_params[max_index[1]]), LinearRegression(),
          RandomForestRegressor(n_estimators=100, random_state=42)]
nazwy = ["Regresja SVR", "Regresja liniowa", "Random Forest"]

# Tworzenie siatki wykresów
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes.flatten()

for i in range(len(modele)):
    model = modele[i]

    # Dopasowanie modelu do danych
    model.fit(X_train, y_train)

    # Przewidywanie wartości
    y_pred = model.predict(X_test)
    print(f"{nazwy[i]}. Przewidywana wartość dla X_test: {y_pred}")

    # Ocena predykcji
    ocen_predykcje(y_pred, y_test)

    # Przedstawienie wyników
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[i])
    axes[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linia idealnych przewidywań
    axes[i].set_xlabel(f"Rzeczywisty czas przeżycia pacjenta \n R2 score: {r2_score(y_test, y_pred)}")
    axes[i].set_ylabel("Predykcja")
    axes[i].set_title(f"{nazwy[i]}")

# Wyświetlenie wszystkich wykresów
plt.tight_layout()  # Lepszy układ
plt.show()

