def operacije(brojevi):
    br_broj=len(brojevi)
    print("Brojevi koje ste unijeli:", brojevi)
    print("Broj unesenih brojeva: ",br_broj)
    print("Srednja vrijednost brojeva:", sum(brojevi)/br_broj)
    print("Minimalna vrijednost brojeva: ", min(brojevi))
    print("Maksimalna vrijednost brojeva: ", max(brojevi))
   

brojevi=[]
br_broj=0
ulaz=0
while(True):
    ulaz=input("Unesite broj ili Done za kraj unosa:")
    if(ulaz=="Done"):
        break
    broj=float(ulaz)
    brojevi.append(broj)
operacije(brojevi)
