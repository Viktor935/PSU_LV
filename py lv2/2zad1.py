import numpy as np
import matplotlib.pyplot as plt

rectangle = np.array([[1,1], [3, 1], [3,2], [2,2], [1,1]])
plt.plot(rectangle[:,0], rectangle[:,1], color='blue')
plt.plot(rectangle[:,0], rectangle[:,1], 'bo')
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.axis([0,4,0,4])
plt.show()