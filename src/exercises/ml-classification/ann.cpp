//ann.cpp 2024
//Linkar com opencv2
#include "procimagem.h"

int main() {
  MNIST mnist(14,true,true);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");

  Mat_<float> ay2(mnist.ay.rows, 10); ay2.setTo(0.0);
  for (int i=0; i<mnist.ay.rows; i++) {
    int j=round(mnist.ay(i)); assert(0<=j && j<10);
    ay2(i,j)=+1.0;
  }

  double t1=tempo();
  Mat_<int> v = (Mat_<int>(1,4) << 196, 100, 30, 10); //196
  CvANN_MLP ind(v); ind.train(mnist.ax, ay2, Mat());

  double t2=tempo();
  Mat_<float> saida; //(1,mnist.qx.cols);
  for (int l=0; l<mnist.nq; l++) {
    ind.predict(mnist.qx.row(l),saida);
    mnist.qp(l)=argMax(saida);
  }
  double t3=tempo();

  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
