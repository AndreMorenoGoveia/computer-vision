#dt2-pele.py
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

ax = np.matrix("4 90; 15 50; 65 70; 5 7; 18 3; 70 80",dtype=np.float32)
ay = np.matrix("1;  2;  0; 1;  2;  0",dtype=np.float32)
qx = np.matrix("20 6;  16 80; 68 8; 80 20; 7 70; 3 50",dtype=np.float32)
qy = np.matrix(" 2; 2; 0; 0; 1; 1",dtype=np.float32)

arvore= tree.DecisionTreeClassifier()
arvore= arvore.fit(ax, ay)
qp=arvore.predict(qx)
print("qp: ", qp)
print("qy: ", np.ravel(qy))

fig=plt.figure(figsize=(8,6))
tree.plot_tree(arvore,filled=True,fontsize=10)
plt.tight_layout()
plt.show()
fig.savefig("results/dt2-pele.png")
