#iris_nn.py
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt

def le(nomearq):
  with open(nomearq,"r") as f:
    linhas=f.readlines()
  linha0=linhas[0].split()
  nl=int(linha0[0]); nc=int(linha0[1])
  a=np.empty((nl,nc),dtype=np.float32)
  for l in range(nl):
    linha=linhas[l+1].split()
    for c in range(nc):
      a[l,c]=np.float32(linha[c])
  return a

### main
ax=le("results/iris_ax.txt"); ay=le("results/iris_ay.txt")
qx=le("results/iris_qx.txt"); qy=le("results/iris_qy.txt")

vizinho = neighbors.KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="brute")
vizinho.fit(ax,ay.ravel())

qp = vizinho.predict(qx)
erros=0;
for i in range(qp.shape[0]):
  if qp[i]!=qy[i]: erros+=1
print("Erros=%d/%d.   Pct=%1.3f%%\n"%(erros,qp.shape[0],100.0*erros/qp.shape[0]))

#https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/
fig=plt.figure(figsize=(12, 12), dpi=80);
for i in range(ax.shape[0]):
  if   ay[i]==0: plt.plot(ax[i,0], ax[i,1], 'ro');
  elif ay[i]==1: plt.plot(ax[i,0], ax[i,1], 'go');
  elif ay[i]==2: plt.plot(ax[i,0], ax[i,1], 'bo');
for i in range(qx.shape[0]):
  if   qy[i]==0: plt.plot(qx[i,0], qx[i,1], 'rx');
  elif qy[i]==1: plt.plot(qx[i,0], qx[i,1], 'gx');
  elif qy[i]==2: plt.plot(qx[i,0], qx[i,1], 'bx');

minx = min( ax[:, 0].min()-1, qx[:, 0].min()-1 )
maxx = max( ax[:, 0].max()+1, qx[:, 0].max()+1 )
miny = min( ax[:, 1].min()-1, qx[:, 1].min()-1 )
maxy = max( ax[:, 1].max()+1, qx[:, 1].max()+1 )

gridx = np.arange(minx, maxx, 0.005)
gridy = np.arange(miny, maxy, 0.002)
xx, yy = np.meshgrid(gridx, gridy)

r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1,r2))
yhat = vizinho.predict(grid)
zz = yhat.reshape(xx.shape)

# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz, cmap='Paired')
plt.tight_layout(); plt.show(); fig.savefig("results/iris_nn.png")
