import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 

import urllib
import xml.etree.ElementTree as ET

mtcars = pd.read_csv('mtcars.csv')

print("--------------------------------------------")
print("1. Kojih 5 automobila ima najveću potrošnju?")
print(mtcars.sort_values(by=['mpg'], ascending = True).head(5))

print("--------------------------------------------------------------")
print("2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju?")
CarCyl8 = mtcars[mtcars.cyl == 8]
print(CarCyl8.sort_values(by=['mpg'], ascending = True).tail(3))

print("---------------------------------------------------------")
print("3. Kolika je srednja potrošnja automobila sa 6 cilindara?")
CarCyl6 = mtcars[mtcars.cyl == 6] 
CarCyl6_avgConsumption = CarCyl6["mpg"].mean()
print(CarCyl6_avgConsumption)

print("-----------------------------------------------------------------------------------")
print("4. Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?")
CarCyl4_mass2000to2200 = mtcars[(mtcars.cyl == 4) & (mtcars.wt>2) & (mtcars.wt<2.2)]
CarCyl4_mass2000to2200_avgConsumption = CarCyl4_mass2000to2200["mpg"].mean()
print(CarCyl4_mass2000to2200_avgConsumption)

print("-----------------------------------------------------------------------------------------")
print("5. Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka?")
gearboxType = mtcars.groupby('am').car
print(gearboxType.count())

print("----------------------------------------------------------------------------------")
print("6. Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?")
Cars_automaticGearbox_over100hp = mtcars[(mtcars.am == 0) & (mtcars.hp>100)]
print(Cars_automaticGearbox_over100hp["car"].count())

print("--------------------------------------------------")
print("7. Kolika je masa svakog automobila u kilogramima?")
pound2kg = 0.45359    # 1 pound = 0.45359 kg
CarIndex = 0
for wt in mtcars["wt"]:   
    wt_kg_value = (wt * 1000) * pound2kg         
    print(str(CarIndex) + ". " + mtcars.car[CarIndex] + ": %.3f kg" % (wt_kg_value))
    CarIndex = CarIndex + 1 