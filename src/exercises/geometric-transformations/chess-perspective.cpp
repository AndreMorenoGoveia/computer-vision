#include "procimagem.h"
int main()
{
    Mat_<float> src = (Mat_<float>(4, 2) << 104, 294,
                       352, 293,
                       138, 44,
                       320, 33);
    Mat_<float> dst = (Mat_<float>(4, 2) << 104, 294,
                       352, 294,
                       104, 33,
                       352, 33);
    Mat_<double> m = getPerspectiveTransform(src, dst);
    cout << m << endl;

    Mat_<double> v = (Mat_<double>(3, 1) << -22, 479, 1);
    Mat_<double> w = m * v;
    cout << w << endl;
    cout << w(0) / w(2) << " " << w(1) / w(2) << endl;

    Mat_<Vec3b> a = imread("assets/calib_result.jpg", 1);
    if (a.total() == 0)
        erro("Erro leitura assets/calib_result.jpg");
    Mat_<Vec3b> b;
    warpPerspective(a, b, m, a.size());
    imwrite("results/calib_aligned.jpg", b);
}
