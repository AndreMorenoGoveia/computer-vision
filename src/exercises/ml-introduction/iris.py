#iris.py
from sklearn import datasets
import numpy as np
import sys

iris = datasets.load_iris()
print(iris)

data=np.array(iris["data"],dtype=np.float32)
with open("assets/irisdata.txt","w") as f:
  print(data.shape[0], data.shape[1], file=f)
  for l in range(data.shape[0]):
    for c in range(data.shape[1]):
      print(data[l,c],end=" ",file=f)
    print(file=f)

target=np.array(iris["target"],dtype=np.float32)
with open("assets/iristarget.txt","w") as f:
  print(target.shape[0], "1", file=f)
  for l in range(target.shape[0]):
    print(target[l],file=f)
