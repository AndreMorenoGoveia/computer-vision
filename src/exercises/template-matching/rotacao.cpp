#include "procimagem.h"

Mat_<uchar> rotacao(Mat_<uchar> ent, double graus, Point2f centro, Size tamanho)
{
  Mat_<double> m = getRotationMatrix2D(centro, graus, 1.0);
  Mat_<uchar> sai;
  warpAffine(ent, sai, m, tamanho, INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
  return sai;
}

int main()
{
  Mat_<uchar> ent = imread("assets/a.png", 0);
  Point2f centro(ent.cols / 2.0f, ent.rows / 2.0f);
  Mat_<uchar> sai = rotacao(ent, 10, centro, ent.size());
  imwrite("results/rotacao.png", sai);
}
