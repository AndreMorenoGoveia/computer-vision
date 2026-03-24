//tabelann.cpp 2024
#include "procimagem.h"

int hamming(int a, int b) {
  assert(0<=a && a<512);
  assert(0<=b && b<512);
  int soma=0;
  int bit=1;
  for (int i=0; i<9; i++) {
    if ( (a & bit) != (b & bit) ) soma++;
    i = i*2;
  }
  return soma;
}

int main(int argc, char** argv) {
  if (argc!=5) erro("tabelann ax.bmp ay.bmp qx.bmp qp.bmp");
  Mat_<uchar> AX=imread(argv[1],0);
  Mat_<uchar> AY=imread(argv[2],0);
  vector<uchar> tabela(512, 128); // Padrao nao visto vira cinza

  //Percorre imagem AX e AY, preenchendo tabela
  for (int l=1; l<AX.rows-1; l++)
    for (int c=1; c<AX.cols-1; c++) {
      int indice=0;
      for (int l2=-1; l2<=1; l2++)
        for (int c2=-1; c2<=1; c2++) {
          int t=1;
          if (AX(l+l2,c+c2)==0) t=0;
          indice = 2*indice+t;
        }
      //Aqui, indice vai ser um numero entre 0 e 511
      tabela[indice]=AY(l,c);
    }

  //Processa as entradas cinza (padroes nao vistos) da tabela.
  for (int indice=0; indice<512; indice++) {
    if (tabela[indice]==128) {
      //procura o vizinho mais proximo na tabela
      int mindist=INT_MAX;
      uchar minsai=0;
      for (int j=0; j<512; j++) {
        if (tabela[j]!=128) {
          int dist=hamming(indice,j);
          if (dist<mindist) {
            mindist=dist;
            minsai=tabela[j];
          }
        }
      }
      tabela[indice]=minsai;
    }
  }

  Mat_<uchar> QX=imread(argv[3],0);
  Mat_<uchar> QP(QX.size(),255);
  for (int l=1; l<QX.rows-1; l++)
    for (int c=1; c<QX.cols-1; c++) {
      int indice=0;
      for (int l2=-1; l2<=1; l2++)
        for (int c2=-1; c2<=1; c2++) {
          int t=1;
          if (QX(l+l2,c+c2)==0) t=0;
          indice = 2*indice+t;
        }
      //Aqui, indice vai ser um numero entre 0 e 511
      QP(l,c)=tabela[indice];
    }
  imwrite(argv[4],QP);
}
