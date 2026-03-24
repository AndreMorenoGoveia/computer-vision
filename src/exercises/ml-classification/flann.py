# flann.py 2024
import tensorflow.keras as keras
from keras.datasets import mnist
import cv2, sys, time
import numpy as np

(AX, ay), (QX, qy) = mnist.load_data();

ax=np.empty((AX.shape[0],14,14))
for i in range(AX.shape[0]): 
  ax[i]=cv2.resize(AX[i],(14,14),cv2.INTER_NEAREST);
ax = ax.reshape(ax.shape[0],ax.shape[1]*ax.shape[2]).astype("float32")/255

qx=np.empty((QX.shape[0],14,14))
for i in range(QX.shape[0]): 
  qx[i]=cv2.resize(QX[i],(14,14),cv2.INTER_NEAREST);
qx = qx.reshape(qx.shape[0],qx.shape[1]*qx.shape[2]).astype("float32")/255

t1 = time.time()
FLANN_INDEX_KDTREE = 1;  # BUG: Faltam "FLANN enums" do OpenCV
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4);
flann = cv2.flann_Index(ax, flann_params)
t2 = time.time()
matches, dists = flann.knnSearch(qx, 1)
t3 = time.time()

# for l in range(matches.shape[0]):
  # i=matches[l]; qp[l]=ay[i]
qp = ay[matches].flatten()

# erros=0;
# for l in range(matches.shape[0]):
  # if qp[l]!=qy[l]: erros+=1
erros = np.count_nonzero(qp!=qy)

print("Erros=%5.2f%%" % (100.0*erros/qy.shape[0]) )
print("Tempo de treinamento: %f"%(t2-t1))
print("Tempo de predicao: %f"%(t3-t2))