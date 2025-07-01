import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


def barrier(x):
    return 2e-23 * np.exp(-np.cos(x) * 50)


x = np.linspace(0, np.pi, 10000)

y = barrier(x)

plt.figure()
plt.plot(x, y)
plt.xlabel('x')
plt.title('Plot of the barrier')
plt.legend()
plt.grid(True)
plt.show()
