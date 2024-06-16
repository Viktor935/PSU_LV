import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 

import urllib
import xml.etree.ElementTree as ET


mtcars = pd.read_csv('mtcars.csv')

# 1. Pomoću barplot-a prikažite na istoj slici potrošnju automobila s 4, 6 i 8 cilindara.
CarCyl4 = mtcars[mtcars.cyl == 4]
CarCyl6 = mtcars[mtcars.cyl == 6]
CarCyl8 = mtcars[mtcars.cyl == 8]

CarCyl4_indexes = np.arange(0, len(CarCyl4), 1)
CarCyl6_indexes = np.arange(0, len(CarCyl6), 1) 
CarCyl8_indexes = np.arange(0, len(CarCyl8), 1) 

widthBetweenColumns = 0.3

plt.figure()

plt.bar(CarCyl4_indexes, CarCyl4["mpg"], widthBetweenColumns, color="r")
plt.bar(CarCyl6_indexes + widthBetweenColumns, CarCyl6["mpg"], widthBetweenColumns, color="g")
plt.bar(CarCyl8_indexes + 2*widthBetweenColumns, CarCyl8["mpg"], widthBetweenColumns, color="b")
plt.title("Potrosnja automobila s 4, 6 i 8 cilindara")
plt.xlabel('auto')
plt.ylabel('mpg')
plt.legend(['4 cilindra','6 cilindara','8 cilindara'], loc=1)
plt.grid(axis='y', linestyle='--')
plt.show()

# 2. Pomoću boxplot-a prikažite na istoj slici distribuciju težine automobila s 4, 6 i 8 cilindara.
CarWeight4=[]
CarWeight6=[]
CarWeight8=[]

for i in CarCyl4["wt"]:
     CarWeight4.append(i) 

for i in CarCyl6["wt"]:
     CarWeight6.append(i)  

for i in CarCyl8["wt"]:
     CarWeight8.append(i)  

plt.figure()
plt.boxplot([CarWeight4, CarWeight6, CarWeight8], positions = [4,6,8]) 
plt.title("Tezina automobila s 4, 6 i 8 cilindara")
plt.xlabel('Broj klipova')
plt.ylabel('Tezina wt')
plt.grid(axis='y',linestyle='--')
plt.show()

# 3. Pomoću odgovarajućeg grafa pokušajte odgovoriti na pitanje imaju li automobili s ručnim mjenjačem veću
# potrošnju od automobila s automatskim mjenjačem?

automaticCars = mtcars[(mtcars.am == 0)]
CarConsumption_automatic=[]
for i in automaticCars["mpg"]:
     CarConsumption_automatic.append(i)
    
manualCars = mtcars[(mtcars.am == 1)]
CarConsumption_manual=[]
for i in manualCars["mpg"]:
     CarConsumption_manual.append(i)

plt.figure()
plt.boxplot([CarConsumption_manual, CarConsumption_automatic], positions = [0,1])
plt.title("Usporedba potrosnja automobila s rucnim, odnosno automatskim mjenjacem")
plt.ylabel('miles per gallon')
plt.xlabel('automatski mjenjac                         rucni mjenjac')
plt.grid(axis='y',linestyle='--')
plt.show()

# 4. Prikažite na istoj slici odnos ubrzanja i snage automobila za automobile s ručnim odnosno automatskim
# mjenjačem. 
AutomaticCar_acceleration=[]
for i in automaticCars["qsec"]:  
     AutomaticCar_acceleration.append(i)
    
AutomaticCar_power=[]  
for i in automaticCars["hp"]:  
     AutomaticCar_power.append(i)
    
ManualCar_acceleration=[]    
for i in manualCars["qsec"]:  
     ManualCar_acceleration.append(i)
    
ManualCar_power=[]    
for i in manualCars["hp"]:  
     ManualCar_power.append(i)

plt.figure()
plt.scatter(AutomaticCar_acceleration, AutomaticCar_power, marker='+')  
plt.scatter(ManualCar_acceleration, ManualCar_power, marker='^', facecolors='none', edgecolors='r')
plt.title("Odnos ubrzanja i snage automobila s rucnim u odnosu na automatski mjenjac")
plt.ylabel('Snaga - hp')
plt.xlabel('Ubrzanje - qsec')
plt.legend(["Automatski mjenjac","Rucni mjenjac"])
plt.grid(linestyle='--')
plt.show()