import numpy as np
import matplotlib.pyplot as plt

def napravi_sliku(velicina_kvadrata, broj_kvadrata_visina, broj_kvadrata_sirina):
    visina_slike = velicina_kvadrata * broj_kvadrata_visina
    sirina_slike = velicina_kvadrata * broj_kvadrata_sirina

    # Inicijalizacija crnog polja za sliku
    crno_polje = np.zeros((visina_slike, sirina_slike))
    #popunjavanje bijelih polja na crnoj pozadini
    for i in range(broj_kvadrata_visina):
        for j in range(broj_kvadrata_sirina):
            if (i + j) % 2 != 0: #uvjet da bijeli kvadrat bude prvi kao u primjeru
                crno_polje[i*velicina_kvadrata:(i+1)*velicina_kvadrata, j*velicina_kvadrata:(j+1)*velicina_kvadrata] = 255
    
    return crno_polje

velicina_kvadrata = 50
broj_kvadrata_visina = 4
broj_kvadrata_sirina = 5

slika = napravi_sliku(velicina_kvadrata, broj_kvadrata_visina, broj_kvadrata_sirina)

plt.imshow(slika, cmap='gray', vmin=0, vmax=255)
plt.show()
