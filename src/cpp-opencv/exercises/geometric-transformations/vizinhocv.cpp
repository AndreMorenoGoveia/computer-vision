// vizinhocv.cpp - 2024
// Especifica fatores de ampliacao
#include "procimagem.h"
int main(int argc, char** argv)
{ if (argc!=5) {
    printf("vizinhocv: Muda resolucao de imagem usando interpolacao vizinho+px.\n");
    printf("vizinhocv ent.pgm sai.pgm fatorl fatorc\n");
    erro("Erro: Numero de argumentos invalido");
  }
  Mat_<uchar> a=imread(argv[1],0); if (a.total()==0) erro("Erro leitura");
  float fatorl,fatorc;
  if (sscanf(argv[3],"%f",&fatorl)!=1) erro("Erro: Leitura fatorl");
  if (sscanf(argv[4],"%f",&fatorc)!=1) erro("Erro: Leitura fatorc");
  int nl=round(a.rows*fatorl);
  int nc=round(a.cols*fatorc);
  Mat_<uchar> b(nl,nc);
  for (int l=0; l<b.rows; l++)
    for (int c=0; c<b.cols; c++)
      b(l,c) = a(round(l/fatorl),round(c/fatorc));
  imwrite(argv[2],b);
}
