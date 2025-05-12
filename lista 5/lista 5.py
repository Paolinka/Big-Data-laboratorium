import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pywt
import time
from scipy.signal import welch


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


# Naprawa struktury przesłanych plików
def naprawa_struktury(dane):
    dane = dane.drop([0, 1])
    dane = dane.rename(columns={'Price' : 'Date'})
    zbadaj_strukture(dane)

    return dane


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
        if col != 'Date':
            dane_imputed[col] = label_encoder.fit_transform(dane[col])

    # Pozostawienie dat
    dane_imputed['Date'] = dane['Date']


    # Wyświetlenie przetworzonych danych
    print("\nPrzetworzone dane:")
    print(dane_imputed.head())

    return dane_imputed


def dwt_feature_extraction(data, wavelet='haar', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    features = []
    for i in range(level+1):
        features.extend(coeffs[i])
    return features


def statystyki_opisowe(dane, plik):
    dane_bez_daty = dane.drop(columns=['Date'])
    kolumny = dane_bez_daty.columns

    print('\n'*2, '-'*10, plik.upper(), '-'*10)

    # Srednia kolumn
    print('\nŚrednie kolumn:\n', dane_bez_daty.mean())

    # Odchylenie standardowe
    print('\nOdchylenie standardowe kolumn:\n', dane_bez_daty.std())

    # Min i max
    print('\nMinimum kolumn:\n', dane_bez_daty.min())
    print('\nMaksimum kolumn:\n', dane_bez_daty.max())

    # Kwantyle
    print('\nKwantyle:\n', dane_bez_daty.quantile())

    

    for kolumna in kolumny:

        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        axes.flatten()

        axes[0].plot(dane['Date'], dane[kolumna])
        axes[0].set_xlabel('Data')
        axes[0].set_ylabel('Wartość')
        axes[0].set_title('Przebieg czasowy')
        axes[0].grid()

        # Różnice między kolejnymi wartościami 
        differences = dane_bez_daty[kolumna].diff()
        axes[1].plot(dane['Date'], differences)
        axes[1].set_xlabel('Data')
        axes[1].set_ylabel('Różnica')
        axes[1].set_title('Różnice między kolejnymi wartościami')
        axes[1].grid()

        dwt_features = dwt_feature_extraction(dane_bez_daty[kolumna])
        axes[2].plot(dwt_features)
        axes[2].set_title('Współczynniki transformacji falkowej')
        axes[2].set_xlabel('Poziom dekompozycji')
        axes[2].set_ylabel('Współczynniki')

        fig.suptitle(f'{plik}: {kolumna}')
        fig.tight_layout()
        plt.show()



# Zadanie 2 i 3 zarazem
pliki = {'lista 5\Bovespa_data_2025-03-31.csv' : 'Bovespa', 'lista 5\CAC40_data_2025-03-31.csv' : 'CAC40', 
         'lista 5\DAX_data_2025-03-31.csv' : 'DAX', 'lista 5\Dow_Jones_Industrial_Average_data_2025-03-31.csv' : 'Dow Jones Industrial'}

for plik, nazwa in pliki.items():
    # Wczytanie danych
    dane = pd.read_csv(plik)
    zbadaj_strukture(dane)

    dane = naprawa_struktury(dane)

    # Obsługa brakujących danych
    dane = obsluga_brakujacych_wartosci(dane, ['Date'])

    # Data jako datetime i sortowanie 
    dane['Date'] = pd.to_datetime(dane['Date'])
    dane = dane.sort_values('Date')

    statystyki_opisowe(dane, nazwa)

# Zadanie 4
def generate_time_series(n=1000000, freq=10):
    t = np.linspace(0, 100, n)
    signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(n)  # Sygnał sinus z szumem
    return pd.Series(signal, index=t)


def widmowa_gestosc_mocy(series):
    freqs, psd = welch(series.values, fs=1.0, nperseg=1024)
    return freqs, psd


def analiza_falowa(series):
    widths = np.arange(1, 128)
    cwt_matrix, _ = pywt.cwt(series.values, widths, 'morl')
    return widths, cwt_matrix


def autokorelacja(series, lag=1):
    return np.corrcoef(series.iloc[:-lag], series.iloc[lag:])[0, 1]

def autokorelacja_pandas(series, lag=1):
    return series.autocorr(lag=lag)

# Generate data
ts = generate_time_series()

# Performance comparison
start_psd_numpy = time.time()
psd_freqs_numpy, psd_values_numpy = widmowa_gestosc_mocy(ts)
end_psd_numpy = time.time()

start_af = time.time()
w, cwt = analiza_falowa(ts)
end_af = time.time()

start_autocorr = time.time()
autocorr_result = autokorelacja(ts, lag=10)
end_autocorr = time.time()

start_pandas_autocorr = time.time()
pandas_autocorr_result = autokorelacja_pandas(ts, lag=10)
end_pandas_autocorr = time.time()

start_dwt = time.time()
feat = dwt_feature_extraction(ts)
end_dwt = time.time()



# Print results 
print(f"Czas obliczeń - WGM NumPy: {end_psd_numpy - start_psd_numpy:.5f} sec")
print(f"Czas obliczeń - analiza falowa NumPy: {end_af - start_af:.5f} sec")
print(f"Czas obliczeń - autokorelacja NumPy: {end_autocorr - start_autocorr:.5f} sec")
print(f"Czas obliczeń - autokorelacja Pandas: {end_pandas_autocorr - start_pandas_autocorr:.5f} sec")
print(f"Czas obliczeń - ekstrakcja cech Pandas: {end_dwt - start_dwt:.5f} sec")
