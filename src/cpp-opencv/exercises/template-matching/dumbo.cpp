// dumbo.cpp - grad2020
#include "procimagem.h"
int main()
{
  Mat_<float> a = imread("assets/figurinhas.jpg", 0);
  a = a / 255;
  Mat_<float> q = imread("assets/dumbo.jpg", 0);
  q = q / 255;
  Mat_<float> q2 = somaAbsDois(dcReject(q));
  Mat_<float> p1 = matchTemplateSame(a, q2, TM_CCORR);
  imwrite("results/dumbo_cc.pgm", 255 * p1);
  Mat_<float> p2 = matchTemplateSame(a, q, TM_CCOEFF_NORMED);
  imwrite("results/dumbo_ncc.pgm", 255 * p2);
}
