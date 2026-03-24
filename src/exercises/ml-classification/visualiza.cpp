//visualiza.cpp - imprime as primeiras 9 imagens de teste de MNIST 2024
#include "procimagem.h"
int main() {
  MNIST mnist(14,true,true); //nao redimensiona imagens
                              //inverte preto/branco=true,
                              //crop bounding box=false
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data");
  mnist.qp.setTo(2); //Coloca uma classificacao errada de proposito
  Mat_<uchar> e=mnist.geraSaidaErros(5,4); //Organiza 10000 imagens em 100 linhas e 100 colunas
  imwrite("results/visualiza.png",e); //Imprime 10000 imagens-teste como visual.png
}
