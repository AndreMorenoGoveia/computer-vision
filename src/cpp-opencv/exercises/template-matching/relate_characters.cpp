#include "procimagem.h"

Mat_<float> rotacao(Mat_<float> ent, double graus, Point2f centro, Size tamanho)
{
    Mat_<double> m = getRotationMatrix2D(centro, graus, 1.0);
    Mat_<float> sai;
    warpAffine(ent, sai, m, tamanho, INTER_LINEAR, BORDER_CONSTANT, Scalar(1.0));
    return sai;
}

void marca(Mat_<Vec3b> &destino, Point2f centro, Size patternSize, double angulo, int indice)
{
    RotatedRect caixa(centro, Size2f(patternSize.width, patternSize.height), angulo);
    Point2f vertices2f[4];
    caixa.points(vertices2f);

    Point vertices[4];
    for (int i = 0; i < 4; i++)
        vertices[i] = vertices2f[i];

    for (int i = 0; i < 4; i++)
        line(destino, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 3);
    circle(destino, centro, 3, Scalar(255, 0, 0), -1);
    putText(destino, to_string(indice), centro + Point2f(6.0f, 6.0f),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, 8);
}

void match_images(Mat_<uchar> ent, int index)
{
    Mat_<Vec3b> resultado;
    cvtColor(ent, resultado, COLOR_GRAY2BGR);

    ent = ent / 255.0;

    return;



    for (int i = 1; i <= 12; i++)
    {
        string path;
        if (i >= 10) {
            path = "assets/extra_q" + to_string(i) + ".jpg";
        }
        else {
            path = "assets/extra_q0" + to_string(i) + ".jpg";
        }

        Mat_<float> pattern = imread(path, 0);
        pattern = pattern / 255.0;

        Point2f centro(pattern.cols / 2.0f, pattern.rows / 2.0f);
        double melhorValor = -DBL_MAX;
        Point melhorPosicao;
        int melhorAngulo = 0;

        for (int angulo = 0; angulo < 360; angulo += 30) {
            Mat_<float> rotacionado = rotacao(pattern, angulo, centro, pattern.size());
            Mat_<float> p = matchTemplateSame(ent, rotacionado, TM_CCOEFF_NORMED);

            double maxVal;
            Point maxLoc;
            minMaxLoc(p, nullptr, &maxVal, nullptr, &maxLoc);
            if (maxVal > melhorValor) {
                melhorValor = maxVal;
                melhorPosicao = maxLoc;
                melhorAngulo = angulo;
            }
        }

        if (melhorValor >= 0.9)
            marca(resultado, Point2f(melhorPosicao.x, melhorPosicao.y), pattern.size(), melhorAngulo, i);
    }
    imwrite("results/resultado_" + to_string(index) + ".jpg", resultado);
}

int main()
{

    for (int i = 1; i <= 8; i++)
    {
        Mat_<uchar> ent = imread("assets/extra_a" + to_string(i) + ".jpg", 0);
        match_images(ent, i);
    }
}
