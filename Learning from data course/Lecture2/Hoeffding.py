from scipy import signal
import numpy as np
import math

def hoeffding (x, epsilon):
	y = 2.0 * math.exp (-2*epsilon*epsilon*x)
	return y

x = np.arange (0, 10, 0.01)

y1 = [hoeffding (x, 0.1) for x in x]
y2 = [hoeffding (x, 0.3) for x in x]
y3 = [hoeffding (x, 0.5) for x in x]
y4 = [hoeffding (x, 0.7) for x in x]
y5 = [hoeffding (x, 0.9) for x in x]

import matplotlib.pyplot as plt

plt.xlabel ('N')
plt.ylabel ('P')

latext2 = r'\varEpsilon'
plt.plot (x, y1, label=r'$%s$ = 0.1' % latext2)
plt.plot (x, y2, label='$%s$ = 0.3' % latext2)
plt.plot (x, y3, label='$%s$ = 0.5' % latext2)
plt.plot (x, y4, label='$%s$ = 0.7' % latext2)
plt.plot (x, y5, label='$%s$ = 0.9' % latext2)

plt.legend ()
plt.show ()