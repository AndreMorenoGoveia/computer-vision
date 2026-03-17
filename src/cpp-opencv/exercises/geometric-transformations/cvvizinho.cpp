//cvvizinho.cpp 2024
#include "procimagem.h"
int main(int argc, char** argv) {
  if (argc!=4) {
    printf("cvvizinho ent.pgm sai.pgm fator\n");
    erro("Erro: Numero de argumentos invalido");
  }
  Mat_<uchar> a=imread(argv[1],0); if (a.total()==0) erro("Erro leitura");
  double fator; sscanf(argv[3],"%lf",&fator);
  Mat_<uchar> b;
  resize(a, b, Size(0,0), fator, fator, INTER_NEAREST);
  imwrite(argv[2],b);
} 