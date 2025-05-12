import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# zadanie 2
tablica = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
minimum = min(tablica)
maksimum = max(tablica)
srednia = np.mean(tablica)
odch_stand = np.std(tablica)

print(tablica, minimum, maksimum, srednia, odch_stand, '\n', sep='\n')

# zadanie 3
tablica2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
element = tablica2d[1][2]
wycinek = tablica2d[:2, 2:]

print(tablica2d, element, wycinek, '\n', sep='\n')

# zadanie 4
print(tablica.reshape(2, 5), np.transpose(tablica.reshape(2, 5)), sep='\n\n')

# zadanie 5
tab1 = np.array([[1, 2, 3], [4, 5, 6]])
tab2 = np.array([[5, 3, 7], [2, 5, 9]])

print('\n', tab1+tab2, tab1*3, sep='\n\n')

# zadanie 6
siatka = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("\nSiatka:")
print('\n', siatka)

wartosci_1d = np.array([10, 20, 30])
siatka_plus = siatka + wartosci_1d
print('\n', siatka_plus)

mnozniki = np.array([2, 3, 4])
siatka_wynikowa = siatka_plus * mnozniki
print('\n', siatka_wynikowa, '\n')

# zadanie 7
random_tab = np.array([random.randint(1, 10) for _ in range(100)])
suma = sum(random_tab)
srednia_rand = np.mean(random_tab)
odch_stand_rand = np.std(random_tab)
skumul_suma = np.cumsum(random_tab)
skumul_iloczyn = np.cumprod(random_tab)

print(random_tab, '\n', suma, srednia_rand, odch_stand_rand, skumul_suma, skumul_iloczyn, sep='\n')

# zadanie 8
unsorted_tab = np.array([random.randint(1, 20) for _ in range(20)])
sorted_tab = np.sort(unsorted_tab)


def wyszukiwanie_binarne(lista, szuk_wartosc):
    if len(lista) == 0:
        print("lista jest pusta!")
    while len(lista) >= 1:
        srodek = round(len(lista) / 2)
        if lista[srodek] == szuk_wartosc:
            print(f"Znaleziono poszukiwaną liczbę.")
            break
        elif len(lista) == 1:
            print("Nie znaleziono liczby")
            break
        elif lista[srodek] > szuk_wartosc:
            lista = lista[:srodek]
        else:
            lista = lista[srodek+1:]


print('\n', unsorted_tab, sorted_tab, sep='\n\n')
wyszukiwanie_binarne(sorted_tab, 10)
print('\n')

# zadanie 9
df = pd.read_csv("waga1.csv", sep=';')
print(df.shape, df.head(), sep='\n')

# zadanie 10
print(df[['plec', 'Wzrost']])
print(df[df['Wzrost'] >= 180])
print(df.loc[df['plec'] == 0])

# zadanie 11
df = df.dropna()
df = df.drop_duplicates()

# zadanie 12
grupa = df.groupby('plec').agg(
    sredni_wzrost=('Wzrost', 'mean'),
    srednia_waga_przed=('Waga_przed', 'mean'),
    srednia_waga_po=('Waga_po', 'mean'),
)
print("Agregowane dane:")
print(grupa)

print(df.describe())

# zadanie 13
df['Zmiana_wagi'] = df['Waga_po'] - df['Waga_przed']
df['Kategoria_Wzrostu'] = df['Wzrost'].apply(lambda x: 'Niski' if x < 165 else 'Średni' if x < 180 else 'Wysoki')
df['Kategoria_Wzrostu'] = df['Kategoria_Wzrostu'].astype(str).str.upper()
print(df.head())

# zadanie 14
grupa.index = ["Mężczyźni", "Kobiety"]

# Tworzenie wykresów z odpowiednimi kolorami
fig, axes = plt.subplots(nrows=len(grupa.columns), ncols=1, figsize=(6, 8))

kolory = {"Mężczyźni": "blue", "Kobiety": "pink"}  # Definiowanie kolorów

for ax, kolumna in zip(axes, grupa.columns):
    grupa[kolumna].plot(kind="bar", ax=ax, color=[kolory[i] for i in grupa.index], legend=True)
    ax.set_title(kolumna)
    ax.set_xlabel("Płeć")
    ax.set_xticklabels(grupa.index, rotation=0)  # Poprawienie etykiet

    # Dodanie legendy z nazwą kolumny + płcią
    legend_labels = [plt.Rectangle((0, 0), 1, 1, color=kolory[i]) for i in grupa.index]
    legend_texts = [f"{kolumna} - {i}" for i in grupa.index]
    ax.legend(legend_labels, legend_texts, title="Legenda", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

# zadanie 15
# Scalanie danych
inne_dane = pd.DataFrame({'plec': [0, 1], 'Opis': ['Mężczyzna', 'Kobieta']})
df = df.merge(inne_dane, on='plec', how='left')
print(df.head())

# Przestawianie danych
pivot_df = df.pivot_table(values='Waga_przed', index='plec', columns='Kategoria_Wzrostu', aggfunc='mean')
print("Przestawione dane:")
print(pivot_df)

# Obsługa danych szeregów czasowych
df['Data'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
print("Dane z datami:")
print(df[['Data', 'Waga_przed', 'Waga_po']].head())

print("Przetworzone dane:")
print(df.head())

# zadanie 16
x = np.linspace(0, 10, 100)  # Generowanie 100 punktów od 0 do 10
y = np.linspace(20, 50, 100)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color="blue", linestyle="--")  # Wykres linii przerywanej

# Dostosowanie wykresu
plt.xlabel("Wartości X")
plt.ylabel("Wartości Y")
plt.title("Wykres liniowy")
plt.grid(True)
plt.show()

# zadanie 17
x2 = np.array([random.randint(1, 20) for _ in range(20)])
y2 = np.array([random.randint(3, 25) for _ in range(20)])

fig, ax = plt.subplots()
ax.scatter(x2, y2)
plt.xlabel("Wartości X")
plt.ylabel("Wartości Y")
plt.title("Wykres punktowy")
plt.grid(True)
plt.show()

# zadanie 18
x3 = ['Wrocław', 'Poznań', 'Kraków', 'Łódź']
y3 = [669564, 541782, 790279, 665259]

fig, ax = plt.subplots()

ax.bar(x3, y3, width=0.4, edgecolor="white", linewidth=0.7, color=['green', 'pink', 'blue', 'yellow'],
       label=["Liczba ludności - " + miasto for miasto in x3])
plt.xlabel("Miasto")
plt.ylabel("Ludność")
plt.title("Wykres słupkowy")
plt.legend(loc="best")
plt.show()

# zadanie 19
dane = np.random.randn(1000)  # 1000 losowych wartości z rozkładu normalnego

plt.figure(figsize=(8, 5))
plt.hist(dane, bins=20, color="purple", edgecolor="black", alpha=0.7)  # Histogram z 20 pojemnikami

# Dostosowanie wykresu
plt.xlabel("Wartości")
plt.ylabel("Częstotliwość")
plt.title("Histogram rozkładu danych")

plt.show()

# zadanie 20
kategorie = ["A", "B", "C", "D"]
wartosci = [30, 20, 40, 10]  # Procentowy udział kategorii

kolory = ["blue", "orange", "green", "red"]  # Kolory segmentów

plt.figure(figsize=(6, 6))
plt.pie(wartosci, labels=kategorie, autopct="%1.1f%%", colors=kolory, wedgeprops={"edgecolor": "black"})

# Dostosowanie wykresu
plt.title("Proporcje kategorii w zestawie danych")
plt.show()

# zadanie 21
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Wykres liniowy
axs[0, 0].plot(x, y, color="blue", linestyle="--")
axs[0, 0].set_title("Wykres liniowy z zad 16")
axs[0, 0].legend()

# Wykres punktowy (scatter)
axs[0, 1].scatter(x2, y2)
axs[0, 1].set_title("Wykres punktowy z zad 17")
axs[0, 1].legend()


axs[1, 0].bar(x3, y3, width=0.4, edgecolor="white", linewidth=0.7, color=['green', 'pink', 'blue', 'yellow'],
       label=["Liczba ludności - " + miasto for miasto in x3])
axs[1, 0].set_title("Wykres słupkowy z zad 18")
axs[1, 0].legend()

# Wykres kwadratowy (parabola)
axs[1, 1].pie(wartosci, labels=kategorie, autopct="%1.1f%%", colors=kolory, wedgeprops={"edgecolor": "black"})
axs[1, 1].set_title("Wykres kołowy z zad 20")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# zadanie 22
# Wybór odpowiednich kolumn do analizy
kolumny_do_wykresu = ["Wzrost", "Waga_przed", "Waga_po"]
df = df[kolumny_do_wykresu]

# Wykres skumulowany (stacked bar)
df_mean = df.mean()
df_mean.plot(kind="bar", stacked=True, color=["blue", "orange", "green"])
plt.title("Skumulowany wykres słupkowy")
plt.xlabel("Cechy")
plt.ylabel("Średnie wartości")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Wykres 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

x = np.arange(len(df))
y = df["Wzrost"]
z = df["Waga_przed"]
ax.scatter(x, y, z, c="red", marker="o", label="Dane")

ax.set_xlabel("Indeks")
ax.set_ylabel("Wzrost")
ax.set_zlabel("Waga przed")
ax.set_title("Wykres 3D")

plt.legend()
plt.show()

# Wykres pudełkowy (boxplot)
plt.figure(figsize=(7, 5))
df.boxplot(column=["Waga_przed", "Waga_po"], patch_artist=True,
           boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red"))
plt.title("Wykres pudełkowy")
plt.ylabel("Wartości")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
