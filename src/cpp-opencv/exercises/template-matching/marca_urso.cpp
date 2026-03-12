// marca_urso.cpp - 2024
#include "procimagem.h"

Mat_<float> preprocessaParaMatching(const Mat_<float> &src)
{
    Mat_<float> borrada;
    GaussianBlur(src, borrada, Size(5, 5), 0.0);

    Mat_<float> gx, gy, mag;
    Sobel(borrada, gx, CV_32F, 1, 0, 3);
    Sobel(borrada, gy, CV_32F, 0, 1, 3);
    magnitude(gx, gy, mag);
    normalize(mag, mag, 0.0, 1.0, NORM_MINMAX);

    return mag;
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
