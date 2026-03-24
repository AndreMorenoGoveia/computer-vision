//bayes.cpp 2024
//Linkar com OpenCV 3/4
#include "procimagem.h"
int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  
  double t1=tempo();

  Ptr<ml::NormalBayesClassifier> ind = ml::NormalBayesClassifier::create();
  ind->train(mnist.ax, ml::ROW_SAMPLE, mnist.AY);  
  double t2=tempo();
  
  Mat_<int> QP(mnist.qy.size());
  ind->predict(mnist.qx,QP);
  for (unsigned i=0; i<mnist.qp.total(); i++) 
    mnist.qp(i)=QP(i);
  double t3=tempo();
  
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
