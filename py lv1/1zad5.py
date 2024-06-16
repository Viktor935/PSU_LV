dat = open("C:\\Users\\student\\Documents\\miki\\lv1\\5z\\song.txt", 'r')


rijecnik = {}


for line in dat:
    line = line.rsplit()
    for rijec in line:
        if rijec in rijecnik:
            rijecnik[rijec] += 1
        else:
            rijecnik[rijec] = 1


unikatne_rijeci = []


for rijec in rijecnik:
    if (rijecnik[rijec] == 1):
        unikatne_rijeci.append(rijec)

print(rijecnik)


print("Broj unikatnih rijeci: ", len(unikatne_rijeci))

print(unikatne_rijeci)