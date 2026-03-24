//dtree34.cpp 2024
//Linkar com OpenCV 3/4
#include "procimagem.h"
int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  TimePoint t1=timePoint();
  CvDTree ind; ind.train(mnist.ax,CV_ROW_SAMPLE,mnist.AY);
  TimePoint t2=timePoint();
  for (int l=0; l<mnist.nq; l++) {
    mnist.qp(l)=ind.predict(mnist.qx.row(l))->value;
  }
  TimePoint t3=timePoint();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treinamento: %f\n",timeSpan(t1,t2));
  printf("Tempo de predicao: %f\n",timeSpan(t2,t3));
}