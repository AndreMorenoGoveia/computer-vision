// marca_urso.cpp - 2024
#include "procimagem.h"

Mat_<float> preprocessaParaMatching(const Mat_<float> &src)
{
    Mat_<float> mx = (Mat_<float>(3, 3) << -3.0, 0.0, +3.0,
                      -10.0, 0.0, +10.0,
                      -3.0, 0.0, +3.0);
    mx = mx / 16.0;
    Mat_<float> my = (Mat_<float>(3, 3) << -3.0, -10.0, -3.0,
                      0.0, 0.0, 0.0,
                      +3.0, +10.0, +3.0);
    my = my / 16.0;

    Mat_<float> gx = filtro2d(src, mx);
    Mat_<float> gy = filtro2d(src, my);

    Mat_<float> gx2;
    pow(gx, 2, gx2);
    Mat_<float> gy2;
    pow(gy, 2, gy2);
    Mat_<float> modgrad;
    pow(gx2 + gy2, 0.5, modgrad);

    return modgrad;
}

Mat_<Vec3f> marca(Mat_<float> a, Mat_<float> p, float limiar)
{
    Mat_<Vec3f> d;
    cvtColor(a, d, COLOR_GRAY2BGR);
    for (int l = 0; l < a.rows; l++)
        for (int c = 0; c < a.cols; c++)
            if (p(l, c) >= limiar)
                circle(d, Point(c, l), 3, Scalar(0.0, 0.0, 1.0), 1, 8, 0);
    return d;
}

int main()
{
    Mat_<float> a = imread("assets/abrinq.pgm", 0);
    a = a / 255.0;
    Mat_<float> q = imread("assets/qbrinq.pgm", 0);
    q = q / 255.0;

    Mat_<float> aPrep = preprocessaParaMatching(a);
    Mat_<float> qPrep = preprocessaParaMatching(q);

    imwrite("results/aprep.png", 255.0 * aPrep);
    
    imwrite("results/qprep.png", 255.0 * qPrep);

    Mat_<float> p = matchTemplateSame(aPrep, qPrep, TM_CCOEFF_NORMED);
    imwrite("results/brinc-template.png", 255.0 * p);
    Mat_<Vec3f> m = marca(a, p, 0.6);
    imwrite("results/brinc.png", 255.0 * m);
}
