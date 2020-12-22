from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import math

e = math.e
rot = e**complex(0, math.pi/18)

#A
# T
#B
TrBA = np.array([
    [rot.real, -rot.imag, 0, 1],
    [rot.imag, rot.real, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
TrAB = np.eye(4)
TrAB[:3, :3] = TrBA[:3, :3].T
TrAB[:3, 3] = -np.dot(TrAB[:3, :3], TrBA[:3, 3])

P1 = [1, 1, 0, 1]
P2 = [0, -0.5, 0, 1]

PAbaxis = np.array([[2,0,0,0],[0,2,0,0]]).T
PAbaxis = np.dot(TrAB, PAbaxis)

PBza = np.dot(TrBA, P1)
print("P1 w ukladzie B")
print(PBza)

print("P2 w ukladzie B")
PAzb = np.dot(TrBA, P2)
print(PAzb)

plt.figure(figsize=(16,9))
plt.plot([0, 1], [0, 0], c=[1,0,0])
plt.plot([0, 0], [0, 1], c=[0.7, 0, 0])
plt.plot(TrAB[[0, 0], -1] + [0, PAbaxis[0, 0]], TrAB[[1,1], -1] + [0, PAbaxis[1, 0]], c=[0,0.9,0])
plt.plot(TrAB[[0, 0], -1] + [0, PAbaxis[0, 1]], TrAB[[1,1], -1] + [0, PAbaxis[1, 1]], c=[0,0.6,0])

#plt.scatter(Pa[0], Pa[1], label="Pa 1,1")
for pt in product(np.linspace(-3, 3, 4), repeat=2):
    pt = [*pt, 0, 1]
    cord_B = np.dot(TrBA, pt)
    plt.scatter(pt[0], pt[1])
    plt.text(pt[0], pt[1]+0.05, f"{cord_B[0]:>3.2f} {cord_B[1]:>3.2f}")
#plt.scatter(P1[0], P1[1], label="P 1")
#plt.text(P1[0], P1[1]+0.2, "Hello")
#plt.scatter(P2[0], P2[1], label="P 2")

#plt.grid()
plt.legend()
bound = 4
#plt.axis([-bound,bound,-bound,bound])
plt.axis('equal')
plt.show()
#plt.show(pause=False)






