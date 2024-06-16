def kategorija_ocijene(ocijena):
    if ocijena>= 0.9 and ocijena <=1.0:
        return "A"
    elif ocijena >= 0.8:
        return "B"
    elif ocijena >= 0.7:
        return "C"
    elif ocijena >= 0.6:
        return "D"
    elif ocijena >= 0.0 and ocijena < 0.6:
        return "F"
    else:
        return "Neispravan unos"

ocijena=-1.0
while(ocijena < 0.0 or ocijena > 1.0):
    ocijena=float(input("Unesite ocijenu: "))
    if ocijena < 0.0 or ocijena > 1.0:
        print("Ocijena mora bit između 0 i 1, unesite ponovno")
       
kategorija=kategorija_ocijene(ocijena)
print("Kategorija ocijene je: ", kategorija)
