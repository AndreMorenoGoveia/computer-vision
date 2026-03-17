# vizinho.py - pos2022
# Especifica fatores de ampliacao

import cv2, sys
import numpy as np
if len(sys.argv)!=5:
  print("vizinho: Muda resolucao de imagem usando interpolacao vizinho+px.")
  print("vizinho ent.pgm sai.pgm fatorl fatorc")
  sys.exit("Erro: Numero de argumentos invalido")

a=cv2.imread(sys.argv[1],0)
if a is None:
  sys.exit("Erro leitura"+sys.argv[1])
fatorl=float(sys.argv[3]); fatorc=float(sys.argv[4])
nl=round(a.shape[0]*fatorl); nc=round(a.shape[1]*fatorc)
b=np.empty((nl,nc),np.uint8)
for l in range(b.shape[0]):
  for c in range(b.shape[1]):
    b[l,c] = a[int(l/fatorl),int(c/fatorc)]
cv2.imwrite(sys.argv[2],b)