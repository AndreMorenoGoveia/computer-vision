//linear.cpp - 2024
#include "procimagem.h"
int main() {
  Mat_<uchar> a=imread("assets/lennag.jpg",0); if (a.total()==0) erro("Erro leitura");
  int nl=round(a.rows*1.5), nc=round(a.cols*1.5);
  Mat_<uchar> b(nl,nc);
  for (int l=0; l<b.rows; l++)
    for (int c=0; c<b.cols; c++) {
      double ald = l * ((a.rows-1.0)/(b.rows-1.0));
      double acd = c * ((a.cols-1.0)/(b.cols-1.0));
      int fal=int(ald); int fac=int(acd);
      double dl=ald-fal; double dc=acd-fac;

      double p1=(1-dl)*(1-dc);
      double p2=(1-dl)*dc;
      double p3=dl*(1-dc);
      double p4=dl*dc;
      b(l,c)= cvRound(
                p1*a(fal,fac)   + p2*a(fal,fac+1)  +
                p3*a(fal+1,fac) + p4*a(fal+1,fac+1)
              );
    }
  imwrite("results/linear.jpg",b);
}
