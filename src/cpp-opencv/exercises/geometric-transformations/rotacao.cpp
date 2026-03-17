//rotacao.cpp 2024
#include "procimagem.h"

inline double deg2rad(double x)
{ return (x/180.0)*(M_PI); }

int main(int argc, char** argv)
{ if (argc!=4) {
    printf("rotacao ent.pgm sai.pgm graus\n");
    erro("Erro: Numero de argumentos invalido");
  }
  double graus; sscanf(argv[3],"%lf",&graus);
  double radianos=deg2rad(graus);
  double co=cos(radianos);
  double se=sin(radianos);

  ImgXyb<uchar> a=imread(argv[1],0); if (a.total()==0) erro("Erro leitura");
  a.centro(a.rows/2,a.cols/2); a.backg=255;
  ImgXyb<uchar> b(a.rows,a.cols);
  b.centro(b.rows/2,b.cols/2); b.backg=255;

  for (int xb=b.minx; xb<=b.maxx; xb++)
    for (int yb=b.miny; yb<=b.maxy; yb++) {
      int xa=cvRound(xb*co+yb*se);
      int ya=cvRound(-xb*se+yb*co);
      b(xb,yb)=a(xa,ya);
    }
  imwrite(argv[2],b);
}
