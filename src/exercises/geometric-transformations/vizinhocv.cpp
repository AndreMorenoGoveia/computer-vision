// vizinhocv.cpp - 2024
// Especifica fatores de ampliacao
#include "procimagem.h"
int main()
{
  Mat_<uchar> a=imread("assets/lennag.jpg",0); if (a.total()==0) erro("Erro leitura");
  float fatorl=1.5, fatorc=1.5;
  int nl=round(a.rows*fatorl);
  int nc=round(a.cols*fatorc);
  Mat_<uchar> b(nl,nc);
  for (int l=0; l<b.rows; l++)
    for (int c=0; c<b.cols; c++)
      b(l,c) = a(round(l/fatorl),round(c/fatorc));
  imwrite("results/vizinhocv.jpg",b);
}
