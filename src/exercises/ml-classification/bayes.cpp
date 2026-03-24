//bayes.cpp 2024
//Linkar com OpenCV2
#include "procimagem.h"
int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  CvNormalBayesClassifier ind;
  ind.train(mnist.ax, mnist.ay);
  double t2=tempo();
  ind.predict(mnist.qx, &mnist.qp);
  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
