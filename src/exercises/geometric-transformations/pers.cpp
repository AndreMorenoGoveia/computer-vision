// pers.cpp grad-2018
#include "procimagem.h"
int main()
{
    Mat_<float> src = (Mat_<float>(4, 2) << 73, 0,
                       533, 0,
                       -22, 479,
                       629, 479);
    Mat_<float> dst = (Mat_<float>(4, 2) << 16, 0,
                       630, 0,
                       14, 479,
                       630, 479);
    Mat_<double> m = getPerspectiveTransform(src, dst);
    cout << m << endl;

    Mat_<double> v = (Mat_<double>(3, 1) << -22, 479, 1);
    Mat_<double> w = m * v;
    cout << w << endl;
    cout << w(0) / w(2) << " " << w(1) / w(2) << endl;

    Mat_<Vec3b> a = imread("assets/ka0.jpg", 1);
    if (a.total() == 0)
        erro("Erro leitura assets/ka0.jpg");
    Mat_<Vec3b> b;
    warpPerspective(a, b, m, a.size());
    imwrite("results/ka1.jpg", b);

    m = m.inv();
    warpPerspective(b, a, m, a.size());
    imwrite("results/ka2.jpg", a);
}
