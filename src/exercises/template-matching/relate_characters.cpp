#include "procimagem.h"

const float DONTCARE = 128.0f / 255.0f;

Size tamanhoRotacaoCompleta(Size original)
{
    int lado = cvRound(ceil(sqrt(original.width * original.width +
                                  original.height * original.height)));
    int largura = lado;
    int altura = lado;

    if ((largura - original.width) % 2 != 0)
        largura++;
    if ((altura - original.height) % 2 != 0)
        altura++;

    return Size(largura, altura);
}

Mat_<float> centralizaEmCanvas(Mat_<float> ent, Size tamanho, float fundo)
{
    Mat_<float> sai(tamanho, fundo);
    int x = (tamanho.width - ent.cols) / 2;
    int y = (tamanho.height - ent.rows) / 2;
    Mat_<float>(sai, Rect(x, y, ent.cols, ent.rows)) = ent;
    return sai;
}

Mat_<float> centralizaMascara(Size original, Size tamanho)
{
    Mat_<float> sai(tamanho, 0.0f);
    int x = (tamanho.width - original.width) / 2;
    int y = (tamanho.height - original.height) / 2;
    Mat_<float>(sai, Rect(x, y, original.width, original.height)) = 1.0f;
    return sai;
}

Mat_<float> rotacao(Mat_<float> ent, double graus, Point2f centro, Size tamanho, float fundo)
{
    Mat_<double> m = getRotationMatrix2D(centro, graus, 1.0);
    Mat_<float> sai;
    warpAffine(ent, sai, m, tamanho, INTER_LINEAR, BORDER_CONSTANT, Scalar(fundo));
    return sai;
}

Mat_<float> rotacaoMascara(Mat_<float> ent, double graus, Point2f centro, Size tamanho)
{
    Mat_<double> m = getRotationMatrix2D(centro, graus, 1.0);
    Mat_<float> sai;
    warpAffine(ent, sai, m, tamanho, INTER_NEAREST, BORDER_CONSTANT, Scalar(0.0f));
    return sai;
}

Rect boundingRectMascara(const Mat_<float> &mascara)
{
    Mat_<uchar> valida = (mascara > 0.5f);
    vector<Point> pontos;
    findNonZero(valida, pontos);
    if (pontos.empty())
        return Rect(0, 0, mascara.cols, mascara.rows);
    return boundingRect(pontos);
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

void match_images(Mat_<float> ent, int index)
{
    Mat_<uchar> ent8u;
    ent.convertTo(ent8u, CV_8U, 255.0);

    Mat_<Vec3b> resultado;
    cvtColor(ent8u, resultado, COLOR_GRAY2BGR);

    imwrite("results/resultado.jpg", resultado);

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

        Size tamanhoRot = tamanhoRotacaoCompleta(pattern.size());
        Mat_<float> patternExpandido = centralizaEmCanvas(pattern, tamanhoRot, DONTCARE);
        Mat_<float> mascaraExpandida = centralizaMascara(pattern.size(), tamanhoRot);
        Point2f centro(tamanhoRot.width / 2.0f, tamanhoRot.height / 2.0f);
        double melhorValor = -DBL_MAX;
        Point melhorPosicao;
        int melhorAngulo = 0;

        for (int angulo = 0; angulo < 360; angulo-=-5) {
            Mat_<float> rotacionado = rotacao(patternExpandido, angulo, centro, tamanhoRot, DONTCARE);
            Mat_<float> mascaraRotacionada = rotacaoMascara(mascaraExpandida, angulo, centro, tamanhoRot);
            Rect roiValida = boundingRectMascara(mascaraRotacionada);
            Mat_<float> rotacionadoRecortado = rotacionado(roiValida).clone();
            Mat_<float> mascaraRecortada = mascaraRotacionada(roiValida).clone();
            Mat_<uchar> pixelsInvalidos = (mascaraRecortada <= 0.5f);
            rotacionadoRecortado.setTo(DONTCARE, pixelsInvalidos);

            Mat_<float> rotacionadoPrep = somaAbsDois(dcReject(rotacionadoRecortado, DONTCARE));
            Mat_<float> p = filtro2d(ent, rotacionadoPrep);

            double maxVal;
            Point maxLoc;
            minMaxLoc(p, nullptr, &maxVal, nullptr, &maxLoc);
            if (maxVal > melhorValor) {
                melhorValor = maxVal;
                melhorPosicao = maxLoc;
                melhorAngulo = angulo;
            }
        }

        marca(resultado, Point2f(melhorPosicao.x, melhorPosicao.y), pattern.size(), melhorAngulo, i);
    }
    imwrite("results/resultado_" + to_string(index) + ".jpg", resultado);
}

int main()
{

    for (int i = 1; i <= 8; i++)
    {
        Mat_<float> ent = imread("assets/extra_a" + to_string(i) + ".jpg", 0);
        ent = ent / 255.0;
        match_images(ent, i);
    }
}
