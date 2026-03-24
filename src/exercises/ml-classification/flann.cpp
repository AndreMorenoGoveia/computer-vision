//flann.cpp 2024
//Linkar com opencv 2, 3 ou 4
#include "procimagem.h"
int main() {
  MNIST mnist(10, true, false);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  flann::Index ind(mnist.ax,flann::KDTreeIndexParams(64));
  double t2=tempo();
  Mat_<int> matches(mnist.na,1); Mat_<float> dists(mnist.na,1);
  ind.knnSearch(mnist.qx, matches, dists, 1, flann::SearchParams(256) );
  for (int l=0; l<mnist.qx.rows; l++) {
    mnist.qp(l)=mnist.ay(matches(l));
  }
  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
