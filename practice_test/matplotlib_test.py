# coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 50)
#plt.plot(x, np.sin(x))
#plt.plot(x, np.sin(x), x, np.sin(2*x))
#plt.plot(x, np.sin(x), 'r-.', x, np.sin(2*x), 'g--o')
plt.subplot(2,2,1) #行，列，活跃区域
plt.plot(x, np.sin(x), 'r-o')
plt.subplot(2,2,2)
plt.plot(x, np.cos(x), 'y-s')
plt.subplot(2,2,3)
plt.plot(x, np.sin(2*x), 'b--^')
plt.show()