def total_euro(radni_h, euro_h):
    total=radni_h*euro_h
    return(total)

radni_h=float(input("Radni h: "))
euro_h=float(input("Euro po sati: "))
ukupno=total_euro(radni_h,euro_h)
print("Ukupno je: ", ukupno)
