import pandas as pd

warzywa = {
    "Pomidor": {"świeże", "czerwone", "słodkie"},
    "Sałata": {"świeże", "zielone", "liściaste"},
    "Papryka": {"świeże", "ostre", "czerwone"},
    "Szpinak": {"mrożone", "zielone", "liściaste"},
    "Marchew": {"świeże", "bulwowe", "słodkie"}
}

preferencje = {
    "A": {"świeże", "ostre", "czerwone"},
    "B": {"mrożone", "zielone", "słodkie", "liściaste"},
    "C": {"świeże", "zielone", "czerwone", "słodkie"}
}


def znajdz_najlepsze_warzywo(preferencje, warzywa):
    wyniki = []
    najlepsze_warzywa = {}

    for klucz, cechy_klienta in preferencje.items():
        dopasowania = {}

        for warzywo, cechy_warzywa in warzywa.items():
            wspolne = cechy_klienta & cechy_warzywa
            dopasowania[warzywo] = len(wspolne)

        najlepsze_warzywo = max(dopasowania, key=dopasowania.get)
        najlepsze_warzywa[klucz] = najlepsze_warzywo

        for warzywo, liczba_cech in dopasowania.items():
            wyniki.append([klucz, warzywo, liczba_cech, ', '.join(warzywa[warzywo] & cechy_klienta)])

    df = pd.DataFrame(wyniki, columns=["Preferencje", "Warzywo", "Liczba dopasowań", "Dopasowane cechy"])
    print(df)

    print("\nNajlepiej dopasowane warzywa dla każdej preferencji:")
    for pref, warzywo in najlepsze_warzywa.items():
        print(f"{pref}: {warzywo}")


znajdz_najlepsze_warzywo(preferencje, warzywa)