#!/usr/bin/env python3
#coding:utf-8
#

import numpy as np
import matplotlib.pyplot as plt

#cm_name = 'jet'
cm_name = 'inferno'

cm = plt.get_cmap(cm_name)

print('colormap(type) = ', type(cm))
print('colormap(N) = ', cm.N)

cm(0) # (R,G,B,alpha)

Rs = []
Gs = []
Bs = []
As = []

for n in range(cm.N):
    Rs.append(cm(n)[0])
    Gs.append(cm(n)[1])
    Bs.append(cm(n)[2])
    As.append(cm(n)[3])


gradient = np.linspace(0,1,cm.N)

gradient_array = np.vstack((gradient, gradient))

fig = plt.figure()

ax = fig.add_axes((0.1,0.3,0.8,0.6))

ax.plot(As, 'k', label='A')
ax.plot(Rs, 'r', label='R')
ax.plot(Gs, 'g', label='G')
ax.plot(Bs, 'b', label='B')
ax.set_xlabel('step')
ax.set_xlim(0, cm.N)
ax.set_title(cm_name)
ax.legend()

ax2 = fig.add_axes((0.1,0.1,0.8,0.05))
ax2 = plt.imshow(gradient_array, aspect='auto', cmap=cm)
#ax2.set_axis_off()
plt.show()
