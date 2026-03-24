#dtree.py - 2024
import numpy as np
from sklearn import tree
import sys, time, cv2
import tensorflow.keras as keras

#<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<
mnist = keras.datasets.mnist
(AX, ay),(QX, qy) = mnist.load_data()

ax=np.empty((AX.shape[0],14,14))
for i in range(AX.shape[0]): 
  ax[i]=cv2.resize(AX[i],(14,14),cv2.INTER_NEAREST);
ax = ax.reshape(ax.shape[0],ax.shape[1]*ax.shape[2]).astype("float32")/255

qx=np.empty((QX.shape[0],14,14))
for i in range(QX.shape[0]): 
  qx[i]=cv2.resize(QX[i],(14,14),cv2.INTER_NEAREST);
qx = qx.reshape(qx.shape[0],qx.shape[1]*qx.shape[2]).astype("float32")/255

t1 = time.time()
arvore= tree.DecisionTreeClassifier()
arvore= arvore.fit(ax, ay)
t2 = time.time()
qp=arvore.predict(qx)
t3 = time.time()

erros = np.sum( qy!=qp )

print("Taxa de erro = %2.3f%%"%(100*erros/qy.shape[0]))
print("Tempo de treinamento: %f"%(t2-t1))
print("Tempo de predicao: %f"%(t3-t2))
