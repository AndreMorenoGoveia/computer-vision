// remove_noise_feb.cpp - 2024
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat_<uchar> mediana(const Mat_<uchar>& a, int raio)
{
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

void processaImagem(const string& entrada, const string& saida, int raio, int iteracoes)
{
  Mat_<uchar> a = imread(entrada, 0);
  if (a.empty())
    throw runtime_error("Nao foi possivel abrir a imagem: " + entrada);

  for (int i = 0; i < iteracoes; i++)
    a = mediana(a, raio);

  imwrite(saida, a);
}

int main()
{
  processaImagem("assets/fever-1.pgm", "results/fever-1-clean.png", 1, 4);
  processaImagem("assets/fever-2.pgm", "results/fever-2-clean.png", 2, 3);
}
