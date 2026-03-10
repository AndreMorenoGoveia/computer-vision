// media_borda.cpp - 2024
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Mat_<uchar> mediamov(Mat_<uchar> a)
{
  Mat_<uchar> b(a.rows, a.cols, uchar(128));
  for (int l = 2; l < b.rows - 2; l++)
    for (int c = 2; c < b.cols - 2; c++)
    {
      int soma = 0;
      for (int l2 = -2; l2 <= 2; l2++)
        for (int c2 = -2; c2 <= 2; c2++)
        {
          int l3 = l + l2;
          int c3 = c + c2;
          soma = soma + a(l3, c3);
        }
      b(l, c) = round(soma / 9.0);
    }
  return b;
}
int main()
{
  Mat_<uchar> a = imread("assets/lion.png", 0);
  Mat_<uchar> b = mediamov(a);
  imwrite("results/media_borda_4.png", b);
}