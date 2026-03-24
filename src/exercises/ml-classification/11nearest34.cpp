//11nearest34.cpp 2024
//Linkar com opencv 3/4 (compila knearest_v3 -ocv -v3 ou compila.sh knearest_v3)
#include "procimagem.h"
int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  Ptr<ml::KNearest>  knn(ml::KNearest::create());
  knn->train(mnist.ax, ml::ROW_SAMPLE, mnist.ay);
  double t2=tempo();

  // Precisamos classificar 10000 imagens de teste mnist.qx (10000x196)
  // Precisamos colocar as 10000 classificacoes em mnist.qp (10000x1)
  Mat_<float> saidas,dists;
  int k=3;
  for (int l=0; l<10000; l++) {
    knn->findNearest(mnist.qx.row(l), k, noArray(), saidas, dists);
    //cout << saidas.rows << " " << saidas.cols << endl;
    //cout << saidas << endl;
    //cout << dists.rows << " " << dists.cols << endl;
    //cout << dists << endl;
    // Calcule a moda (o elemento mais frequente) de saidas.
    // Coloque moda em mnist.qp(l)
    Mat_<int> freq(1,10,int(0));
    for (int i=0; i<saidas.total(); i++)
      freq(saidas(i))++;
    int indice=argMax(freq);
    mnist.qp(l)=indice;
  }

  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
