import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import splrep, splev, CubicHermiteSpline, CubicSpline, interp1d
from scipy.signal import argrelextrema


def generate_weather_data(num_stations, num_days):
    """
    Funkcja generuje przykładowe dane meteorologiczne dla wielu stacji pomiarowych i dni i zapisuje je do pliku CSV.
    Parametry:
    - num_stations: liczba stacji pomiarowych
    - num_days: liczba dni pomiarowych
    Zwraca:
    - None
    """
    # Temperatury miesięczne dla stacji 1
    temperatures1 = np.array([-2, 0, 5, 12, 18, 23, 26, 25, 21, 15, 8, 2])

    # Generowanie danych dla stacji 1
    np.random.seed(0)
    dates = pd.date_range(start='2023-01-01', periods=num_days)
    station_ids = ['Station_' + str(i) for i in range(1, num_stations + 1)]
    data = {station: [] for station in station_ids}

    for day in range(num_days):
        month = dates[day].month - 1 # Indeksowanie od zera
        temperature1 = temperatures1[month]

        # Generowanie danych dla pozostałych stacji z odchyłkami
        for station in station_ids:
            temperature = temperature1 + np.random.uniform(low=-2, high=2) if station == 'Station_1' else temperature1 + np.random.uniform(low=-4, high=4)
            if day > 0 and np.random.rand() < 0.05: # Rzadkie skoki temperatury
                temperature += np.random.uniform(low=-10, high=10)
            data[station].append(temperature)

    # Utworzenie ramki danych
    df = pd.DataFrame(data)
    df['Date'] = dates
    df = df[['Date'] + station_ids]

    # Zapisanie danych do pliku CSV
    df.to_csv('lista 4\weather_data.csv', index=False)


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


def obsluga_brakujacych_wartosci(dane, kolumny_tekstowe = []):
    # Identyfikacja brakujących wartości i imputacja

    # Przekonwertowanie danych na numeryczne, aby umożliwić imputację
    if kolumny_tekstowe:
        dane_numeric = dane.drop(columns=kolumny_tekstowe) # Usunięcie kolumny z danymi tekstowymi
    else:
        dane_numeric = dane
    imputer = SimpleImputer(strategy='mean')
    dane_imputed = pd.DataFrame(imputer.fit_transform(dane_numeric), columns=dane_numeric.columns)

    # Zastosowanie kodowania zmiennych kategorycznych
    label_encoder = LabelEncoder()
    categorical_columns = dane.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dane_imputed[col] = label_encoder.fit_transform(dane[col])


    # Wyświetlenie przetworzonych danych
    print("\nPrzetworzone dane:")
    print(dane_imputed.head())

    return dane_imputed


def interpolacja_B_sklejana(x_values, y_values, k=3):
    tck = splrep(x_values, y_values, k=k) # k oznacza stopień krzywej Bsklejanej (tutaj wielomian trzeciego stopnia)
    
    y_interpolated = splev(x_values, tck)

    return y_interpolated


def polynomial_interpolation(x_values, y_values, x, degree=4):
    """
    Funkcja wykonuje interpolację wielomianową dla danych punktów.
    Parametry:
    - x_values: tablica NumPy zawierająca wartości x dla znanych punktów danych
    - y_values: tablica NumPy zawierająca odpowiadające wartości y dla znanych
    punktów danych
    - x: wartość x, dla której ma zostać przewidziana wartość y
    Zwraca:
    - y: przewidywana wartość y dla podanej wartości x
    """
    # Wyznaczenie współczynników wielomianu interpolacyjnego
    coefficients = np.polyfit(x_values, y_values, degree)

    # Wyliczenie wartości y dla podanej wartości x
    y = np.polyval(coefficients, x)
    return y


def cubic_hermite_interpolation(x_values, y_values, y_derivatives, x):
    """
    Funkcja wykonuje interpolację kubiczną Hermite'a dla danych punktów.
    Parametry:
    - x_values: tablica NumPy zawierająca wartości x dla znanych punktów danych
    - y_values: tablica NumPy zawierająca odpowiadające wartości y dla znanych
    punktów danych
    - y_derivatives: tablica NumPy zawierająca pochodne pierwszego rzędu
    wartości y w punktach danych
    - x: wartość x, dla której ma zostać przewidziana wartość y
    Zwraca:
    - y: przewidywana wartość y dla podanej wartości x
    """
    # Utworzenie obiektu interpolacyjnego kubicznego Hermite'a
    spline = CubicHermiteSpline(x_values, y_values, y_derivatives)

    # Interpolacja wartości y dla podanej wartości x
    y = spline(x)
    return y, spline


def cubic_spline_interpolation(x_values, y_values, x):
    """
    Funkcja wykonuje interpolację krzywych sklejanych dla danych punktów.
    Parametry:
    - x_values: tablica NumPy zawierająca wartości x dla znanych punktów danych
    - y_values: tablica NumPy zawierająca odpowiadające wartości y dla znanych
    punktów danych
    - x: wartość x, dla której ma zostać przewidziana wartość y
    Zwraca:
    - y: przewidywana wartość y dla podanej wartości x
    """
    # Utworzenie obiektu interpolacyjnego krzywych sklejanych
    spline = CubicSpline(x_values, y_values)

    # Interpolacja wartości y dla podanej wartości x
    y = spline(x)
    return y


# Zadanie 2
ilość_stacji = 6
ilość_dni = 365
generate_weather_data(num_stations=ilość_stacji, num_days=ilość_dni)

data = pd.read_csv('lista 4\weather_data.csv')
zbadaj_strukture(data)
new_data = obsluga_brakujacych_wartosci(data, ['Date'])


# Zadanie 3
ilość_wierszy = ilość_stacji//2 if ilość_stacji % 2 == 0 else ilość_stacji//2 + 1

fig, axes = plt.subplots(ilość_wierszy, 2, figsize=(12, 6))
axes = axes.flatten()  # Spłaszczenie dla łatwego iterowania

for i in range(ilość_stacji):
    x = new_data['Date']
    y = new_data['Station_' + str(i+1)]
    y_inter = interpolacja_B_sklejana(x, y)

    axes[i].scatter(x, y, color='blue', label='Dane')
    axes[i].plot(x, y_inter, color='red', label='Interpolacja B-sklejana')
    axes[i].set_xlabel('Dzień roku')
    axes[i].set_ylabel('Temperatura')
    axes[i].set_title('Stacja' + str(i+1))
    axes[i].get_legend()

plt.suptitle('Interpolacja B-sklejana danych temperatury')
plt.tight_layout()
plt.show()


# Dodaję kolumnę z miesiącem
data["Date"] = pd.to_datetime(data["Date"])
data["Month"] = data["Date"].dt.month

# Obliczam średnią dla miesiąca
monthly_avg = data.groupby("Month").mean()

fig, axes = plt.subplots(ilość_wierszy, 2, figsize=(12, 6))
axes = axes.flatten()  # Spłaszczenie dla łatwego iterowania

for i in range(ilość_stacji):
    x = monthly_avg.index
    y = monthly_avg['Station_' + str(i+1)]
    y_inter = interpolacja_B_sklejana(x, y)

    axes[i].scatter(x, y, color='blue', label='Dane')
    axes[i].plot(x, y_inter, color='red', label='Interpolacja B-sklejana')
    axes[i].set_xlabel('Miesiąc')
    axes[i].set_ylabel('Temperatura')
    axes[i].set_title('Stacja' + str(i+1))
    axes[i].get_legend()

plt.suptitle('Interpolacja B-sklejana danych średniej temperatury')
plt.tight_layout()
plt.show()


# Zadanie 4
# Wczytanie danych
energy_data = pd.read_csv('lista 3/temperature_and_energy_consumption.csv')
zbadaj_strukture(energy_data)
energy_data = obsluga_brakujacych_wartosci(energy_data, ['time_n'])
energy_data = energy_data.sort_values("time_n")


# Dane do interpolacji
x_values = energy_data['time_n'][:100]
y_values = energy_data['energy_consumption'][:100]

# Interplacja wielomianowa
stopien_wielomianu = len(x_values) - 1
x_interpolated = 60 # Wartość x do interpolacji
y_interpolated = polynomial_interpolation(x_values, y_values, x_interpolated, stopien_wielomianu)
print("Przewidywana wartość y dla x =", x_interpolated, ":", y_interpolated)

# Interpolacja kubiczna Hermite'a
# Obliczenie pochodnych 
y_derivatives = np.gradient(y_values, x_values)
y_interpolated_hermite, spline = cubic_hermite_interpolation(x_values, y_values, y_derivatives, x_interpolated)


# Wykres
plt.scatter(x_values, y_values, color='blue', label='Dane')
plt.plot(x_values, y_values, color='blue', linestyle='dashed', label='Interpolacja liniowa')
plt.scatter(x_interpolated, y_interpolated, color='red', label=f'Interpolacja wielomianowa dla x={x_interpolated}')
plt.scatter(x_interpolated, y_interpolated_hermite, color='purple', label=f'Interpolacja kubiczna Hermite\'a dla x={x_interpolated}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja wielomianowa danych')
plt.legend()
plt.tight_layout()
plt.show()

# wniosek: interpolacja kubiczna Hermite'a lepiej działa


# Zadanie 5

stock_data = pd.read_csv('lista 4\AAPL_stock_data.csv')

# zbadanie struktury danych
zbadaj_strukture(stock_data)

# Obsługa brakujących wartości
stock_data = obsluga_brakujacych_wartosci(stock_data, ['Date'])
stock_data = stock_data.sort_values("Date")

# Dane
x = stock_data['Date']
y = stock_data['Close']

# Obliczenie pochodnych 
y_derivatives = np.gradient(y, x)


# Interpolacja kubiczna Hermite'a
x_interpolated = np.linspace(min(x), max(x), 100)
y_interpolated, spline = cubic_hermite_interpolation(x, y, y_derivatives, x_interpolated)

# Identyfikacja lokalnych ekstremów
extrema_max = argrelextrema(y_interpolated, np.greater)[0]
extrema_min = argrelextrema(y_interpolated, np.less)[0]

# Druga pochodna
y_second_derivative = spline.derivative(nu=2)(x_interpolated)

# Identyfikacja punktów przegięcia (zmiana drugiej pochodnej)
# zmiana znaku drugiej pochodnej sugeruje punkt przegięcia (zmiana kierunku trendu).
inflection_points = np.where(np.diff(np.sign(y_second_derivative)))[0]

# Wykresy
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.scatter(x, y, color='blue', label='Dane')
plt.plot(x_interpolated, y_interpolated, color='red', label='Interpolacja kubiczna Hermite\'a')
plt.xlabel('Dzień')
plt.ylabel('Cena zamknięcia')
plt.title('Interpolacja kubiczna Hermite\'a danych')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x_interpolated, y_interpolated, color='red', label='Interpolacja kubiczna Hermite\'a')
plt.scatter(x_interpolated[extrema_max], y_interpolated[extrema_max], color='red', label="Lokalne maksima")
plt.scatter(x_interpolated[extrema_min], y_interpolated[extrema_min], color='green', label="Lokalne minima")
plt.xlabel('Dzień')
plt.ylabel('Cena zamknięcia')
plt.title('Interpolacja kubiczna Hermite\'a - ekstrema')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x_interpolated, y_interpolated, color='red', label='Interpolacja kubiczna Hermite\'a')
plt.scatter(x_interpolated[inflection_points], y_interpolated[inflection_points], color='purple', label="Punkty przegięcia", marker='x')
plt.xlabel('Dzień')
plt.ylabel('Cena zamknięcia')
plt.title('Interpolacja kubiczna Hermite\'a - pkty przegięcia')
plt.legend()

plt.suptitle("Apple stock data")
plt.tight_layout()
plt.show()


# Zadanie 6
# Wczytanie danych
traffic_data = pd.read_csv('lista 4\\road_traffic.csv')
zbadaj_strukture(traffic_data)
traffic_data = obsluga_brakujacych_wartosci(traffic_data, ['Local Time (Sensor)', 'Date', 'Time', 'countlineName', 'direction'])

# Usuwanie niepotrzebnych kolumn z ramki
traffic_data = traffic_data.drop(columns=['OGV1', 'OGV2', 'LGV', 'Time', 'Date', 'countlineName', 'direction'])

# Sortowanie i grupowanie po 'Local Time (Sensor)' oraz sumowanie wartości
traffic_data = traffic_data.sort_values('Local Time (Sensor)')

traffic_data = traffic_data.groupby('Local Time (Sensor)').sum().reset_index()


x_values = traffic_data['Local Time (Sensor)'][:100]
vehicles = ['Car', 'Pedestrian', 'Cyclist', 'Motorbike', 'Bus']

for i, vehicle in enumerate(vehicles):
    y_values = traffic_data[vehicle][:100]
 
    y_inter_B = interpolacja_B_sklejana(x_values, y_values)

    stopien_wielomianu = len(x_values) - 1
    x_interpolated = x_values
    y_interpolated = polynomial_interpolation(x_values, y_values, x_interpolated, stopien_wielomianu)

    # Interpolacja kubiczna Hermite'a
    # Obliczenie pochodnych 
    y_derivatives = np.gradient(y_values, x_values)
    y_interpolated_hermite, spline = cubic_hermite_interpolation(x_values, y_values, y_derivatives, x_interpolated)

    # Interpolacja metodą najbliższego sąsiada
    interpolator = interp1d(x_values, y_values, kind='nearest')

    # Interpolacja dla nowych wartości x
    y_interpolated_KNN = interpolator(x_interpolated)

    
    plt.scatter(x_values, y_values, alpha=0.7, color='lightblue',edgecolors='blue', label='Dane')
    plt.plot(x_values, y_inter_B, color='red', label='Interpolacja B-sklejana')
    plt.plot(x_interpolated, y_interpolated, color='green', label='Interpolacja wielomianowa')
    plt.plot(x_interpolated, y_interpolated_hermite, color='magenta', label='Interpolacja kubiczna Hermite\'a')
    plt.plot(x_interpolated, y_interpolated_KNN, color='orange', label='Interpolacja metodą najbl. sasiada')
    plt.xlabel('Local time')
    plt.ylabel('Traffic')
    plt.title(f'{vehicle}')
    plt.legend()

    plt.suptitle("Porównanie interpolacji")
    plt.tight_layout()
    plt.show()

