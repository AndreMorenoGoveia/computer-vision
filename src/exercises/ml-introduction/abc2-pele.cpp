//abc2-pele.cpp
#include "procimagem.h"

float distancia(Mat_<float> a, Mat_<float> b) {
  if (a.total()!=b.total()) { fprintf(stderr,"Erro distancia"); exit(0); }
  return norm(a,b);
}

int main() {
  Mat_<float> ax = (Mat_<float>(6,2) <<  4, 90, 15, 50, 65, 70, 5, 7, 18, 3, 70, 80);
  Mat_<float> ay = (Mat_<float>(6,1) <<  1,  2,  0, 1,  2,  0);
  Mat_<float> qx = (Mat_<float>(6,2) << 20, 6,  16, 80, 68, 8, 80, 20, 7, 70, 3, 50);
  Mat_<float> qy = (Mat_<float>(6,1) <<  2, 2, 0, 0, 1, 1);
  Mat_<float> qp(6,1);

  for (int iq=0; iq<qx.rows; iq++) {
    float menorDist=FLT_MAX, menorAy;
    for (int ia=0; ia<ax.rows; ia++) {
      float dist=distancia(ax.row(ia),qx.row(iq));
      if (dist<menorDist) { menorDist=dist; menorAy=ay(ia); }
    }
    qp(iq)=menorAy;
  }
  cout << "qp:\n" << qp << endl;
  cout << "qy:\n" << qy << endl;
}

