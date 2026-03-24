//knearest.cpp 2024
//Linkar com opencv2 (compila knearest -ocv -v2)
#include "procimagem.h"
int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  CvKNearest ind(mnist.ax,mnist.ay,Mat(),false,1);
  double t2=tempo();
  ind.find_nearest(mnist.qx, 1, &mnist.qp);
  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
