#naivebayes.py - grad2023
import numpy as np
from sklearn.naive_bayes import GaussianNB
import sys, cv2

#<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<
ax=cv2.imread("assets/ax.png",1)
ay=cv2.imread("assets/ay.png",0)
if ax.shape[0]!=ay.shape[0] or ax.shape[1]!=ay.shape[1]:
  sys.exit("Erro: Imagens com dimensoes diferentes");

features=(ax/255).reshape((ax.shape[0]*ax.shape[1],3))
saidas  =(ay/255).reshape((ay.shape[0]*ax.shape[1],))

gnb = GaussianNB()
modelo = gnb.fit(features, saidas)

qx = cv2.imread("assets/f1.jpg",1)
query = (qx/255).reshape((qx.shape[0]*qx.shape[1],3))
qp = modelo.predict(query)
qp = qp.reshape((qx.shape[0],qx.shape[1]))
cv2.imwrite("results/f1-naivebayes.png",255*qp)
