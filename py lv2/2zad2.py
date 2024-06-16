
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),delimiter=",", skiprows=1)
plt.scatter(data[:,0], data[:,3], s=20, c='red', marker='s')
plt.scatter(data[:,0], data[:,5], s=data[:,5]*5, c='blue', marker='o')
print(data[:,5])
plt.xlabel('mpg')
plt.ylabel('hp(crvena), wt(plava)')
plt.title('Ovisnost potrosnje automobila')
if np.any(data[:,1]==6):
    uvjet=data[data[:, 1] == 6, 0]
    avrage=np.mean(uvjet)
    avrage=np.round(avrage,2)
    minimum=min(uvjet)
    maximum=max(uvjet)
print(avrage, minimum, maximum)
plt.show()