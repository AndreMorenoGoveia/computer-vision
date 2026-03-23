//cvvizinho.cpp 2024
#include "procimagem.h"
int main() {
  Mat_<uchar> a=imread("assets/lennag.jpg",0); if (a.total()==0) erro("Erro leitura");
  double fatorLinha=1.5;
  double fatorColuna=2;
  Mat_<uchar> b;
  resize(a, b, Size(0,0), fatorLinha, fatorColuna, INTER_NEAREST);
  imwrite("results/cvvizinho.jpg",b);
} 
