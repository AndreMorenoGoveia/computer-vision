// mediana.cpp - 2024
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat_<uchar> mediana(Mat_<uchar> a)
{
  const int raio = 2;
  Mat_<uchar> b(a.rows, a.cols);
  vector<int> v;
  for (int l = 0; l < b.rows; l++)
    for (int c = 0; c < b.cols; c++)
    {
      v.resize(0);
      for (int l2 = -raio; l2 <= raio; l2++)
        for (int c2 = -raio; c2 <= raio; c2++)
        {
          int l3 = l + l2;
          int c3 = c + c2;
          if (l3 < 0)
            l3 = -l3;
          if (a.rows <= l3)
            l3 = a.rows - (l3 - a.rows + 2);
          if (c3 < 0)
            c3 = -c3;
          if (a.cols <= c3)
            c3 = a.cols - (c3 - a.cols + 2);
          v.push_back(a(l3, c3));
        }
      auto meio = v.begin() + v.size() / 2;
      nth_element(v.begin(), meio, v.end());
      b(l, c) = *meio;
    }
  return b;
}

int main()
{
  Mat_<uchar> a = imread("assets/fever-2.pgm", 0);

  for (int i = 0; i < 3; i++)
    a = mediana(a);

  imwrite("results/fever-2-clean.png", a);
}
