# Sources: https://gist.github.com/CountChu/097b8c1193ac1a31f9b76d6fadcdac63
from scipy import signal
import numpy as np
import math

def hoeffding (x, epsilon):
	y = 2.0 * math.exp (-2*epsilon*epsilon*x)
	return y

#x = np.arange (0, 100, 0.01)
x = np.arange (0, 10, 0.01)

y1 = [hoeffding (x, 0.1) for x in x]
y2 = [hoeffding (x, 0.3) for x in x]
y3 = [hoeffding (x, 0.5) for x in x]
y4 = [hoeffding (x, 1) for x in x]

import matplotlib.pyplot as plt

plt.xlabel ('N')
plt.ylabel ('P')
latex1 = r'P\leq2e^{\left( -2\varepsilon ^{2}N\right)}'
plt.title (r"Hoeffding's Inequality: $ %s $" % latex1)

latext2 = r'\varepsilon'
plt.plot (x, y1, label=r'$%s$ = 0.1' % latext2)
plt.plot (x, y2, label='$%s$ = 0.3' % latext2)
plt.plot (x, y3, label='$%s$ = 0.5' % latext2)
plt.plot (x, y4, label='$%s$ = 1' % latext2)

plt.legend ()
plt.show ()