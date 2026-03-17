//nn_man.cpp
#include "procimagem.h"
int main() {
  Mat_<float> ax; le(ax,"results/banknote_ax.txt");
  Mat_<float> ay; le(ay,"results/banknote_ay.txt");
  Mat_<float> qx; le(qx,"results/banknote_qx.txt");
  Mat_<float> qy; le(qy,"results/banknote_qy.txt");
  Mat_<float> qp(qy.rows,qy.cols);
  for (int iq=0; iq<qx.rows; iq++) {
    float menorDist=FLT_MAX, menorAy;
    for (int ia=0; ia<ax.rows; ia++) {
      float dist=norm(ax.row(ia)-qx.row(iq));
      if (dist<menorDist) { menorDist=dist; menorAy=ay(ia); }
    }
    qp(iq)=menorAy;
  }
  int erros=0;
  for (int i=0; i<qp.rows; i++)
    if (qp(i)!=qy(i)) erros++;
  printf("Erros=%d/%d.   Pct=%1.3f%%\n",erros,qp.rows,100.0*erros/qp.rows);
}
