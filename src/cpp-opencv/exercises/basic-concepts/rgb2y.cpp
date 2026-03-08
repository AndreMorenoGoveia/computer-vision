// rgb2y.cpp
#include <cekeikon.h>

int main()
{
    Mat_<COR> a;
    le(a, "assets/mandrill.jpg");
    Mat_<GRY> gcorreto(a.size());
    Mat_<GRY> gmedia(a.size());
    Mat_<GRY> gopencv;
    for (unsigned i = 0; i < a.total(); i++)
    {
        gcorreto(i) = round(0.299 * a(i)[2] + 0.587 * a(i)[1] + 0.114 * a(i)[0]);
        gmedia(i) = round((a(i)[0] + a(i)[1] + a(i)[2]) / 3.0);
    }
    cvtColor(a, gopencv, CV_BGR2GRAY);
    imp(gcorreto, "results/gcorreto.bmp");
    imp(gmedia, "results/gmedia.bmp");
    imp(gopencv, "results/gopencv.bmp");
}
