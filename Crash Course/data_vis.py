import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.arange(0, 10)
y = x**2
plt.plot(x, y, 'b--')
plt.xlim(0, 4)
plt.ylim(0, 10)
plt.title("Hello")
plt.xlabel("yolo")
plt.ylabel("gucci")
plt.show()


mat = np.arange(0, 100).reshape(10, 10)
print mat
plt.imshow(mat, cmap = 'GnBu')
plt.colorbar()
plt.show()