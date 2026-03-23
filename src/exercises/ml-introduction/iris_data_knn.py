#iris_nn.py
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import tree

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

def separa_treino_teste(x, y):
  treino = np.arange(x.shape[0]) % 2 == 0
  teste = ~treino
  return x[treino], y[treino], x[teste], y[teste]

def avalia_classificador(classificador, ax, ay, qx, qy):
  classificador.fit(ax, ay.ravel())
  qp = classificador.predict(qx)
  erros = np.count_nonzero(qp != qy.ravel())
  return erros, qp.shape[0], 100.0 * erros / qp.shape[0]

### main
irisdata=le("assets/irisdata.txt")
iristarget=le("assets/iristarget.txt")
ax, ay, qx, qy = separa_treino_teste(irisdata, iristarget)

vizinho = neighbors.KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="brute")
erros_knn, total_knn, pct_knn = avalia_classificador(vizinho, ax, ay, qx, qy)

arvore = tree.DecisionTreeClassifier()
erros_dt, total_dt, pct_dt = avalia_classificador(arvore, ax, ay, qx, qy)

logreg = linear_model.LogisticRegression(max_iter=1000)
erros_lr, total_lr, pct_lr = avalia_classificador(logreg, ax, ay, qx, qy)

svc = svm.SVC()
erros_svm, total_svm, pct_svm = avalia_classificador(svc, ax, ay, qx, qy)

random_forest = ensemble.RandomForestClassifier()
erros_rf, total_rf, pct_rf = avalia_classificador(random_forest, ax, ay, qx, qy)

boost = ensemble.AdaBoostClassifier()
erros_boost, total_boost, pct_boost = avalia_classificador(boost, ax, ay, qx, qy)

print("KNN (k=1): Erros=%d/%d.   Porcentagem de erro=%1.3f%%" % (erros_knn, total_knn, pct_knn))
print("Arvore de decisao: Erros=%d/%d.   Porcentagem de erro=%1.3f%%" % (erros_dt, total_dt, pct_dt))
print("Regressao logistica: Erros=%d/%d.   Porcentagem de erro=%1.3f%%" % (erros_lr, total_lr, pct_lr))
print("SVM: Erros=%d/%d.   Porcentagem de erro=%1.3f%%" % (erros_svm, total_svm, pct_svm))
print("Random forest: Erros=%d/%d.   Porcentagem de erro=%1.3f%%" % (erros_rf, total_rf, pct_rf))
print("Boost: Erros=%d/%d.   Porcentagem de erro=%1.3f%%" % (erros_boost, total_boost, pct_boost))
