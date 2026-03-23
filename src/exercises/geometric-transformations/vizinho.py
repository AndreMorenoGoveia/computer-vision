# vizinho.py - pos2022
# Especifica fatores de ampliacao

import cv2
import numpy as np

a=cv2.imread("assets/lennag.jpg",0)
if a is None:
  raise SystemExit("Erro leitura assets/lennag.jpg")
fatorl=1.5; fatorc=1.5
nl=round(a.shape[0]*fatorl); nc=round(a.shape[1]*fatorc)
b=np.empty((nl,nc),np.uint8)
for l in range(b.shape[0]):
  for c in range(b.shape[1]):
    b[l,c] = a[int(l/fatorl),int(c/fatorc)]
cv2.imwrite("results/vizinho.jpg",b)
