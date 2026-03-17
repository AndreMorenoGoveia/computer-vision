//abc_cv.cpp
#include "procimagem.h"
using namespace std;
using namespace cv;
int main() {
  Mat_<float> ax = (Mat_<float>(6,1) <<  4, 15, 65, 5, 18, 70);
  Mat_<float> ay = (Mat_<float>(6,1) <<  1,  2,  0, 1,  2,  0);
  Mat_<float> qx = (Mat_<float>(3,1) << 16,  3, 75);
  Mat_<float> qy = (Mat_<float>(3,1) <<  2,  1,  0);
  Mat_<float> qp(3,1);

  for (int iq=0; iq<qx.rows; iq++) {
    float menorDist=FLT_MAX, menorAy;
    for (int ia=0; ia<ax.rows; ia++) {
      float dist=abs(ax(ia)-qx(iq));
      if (dist<menorDist) { menorDist=dist; menorAy=ay(ia); }
    }
    qp(iq)=menorAy;
  }
  cout << "qp:\n" << qp << endl;
  cout << "qy:\n" << qy << endl;
}
