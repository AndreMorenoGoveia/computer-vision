#include <cekeikon.h>

Vec3b amarelo(20, 200, 200);

void pintaVermelho(Mat_<COR> &a, int l, int c)
{
    if (distancia(amarelo, a(l, c)) < 100)
    {
        a(l, c) = COR(0, 0, 255);
        pintaVermelho(a, l - 1, c);
        pintaVermelho(a, l, c - 1);
        pintaVermelho(a, l + 1, c);
        pintaVermelho(a, l, c + 1);
    }
}
int main()
{
    Mat_<COR> a;
    le(a, "assets/elefante.jpg");
    pintaVermelho(a, 138, 120);
    imp(a, "results/elefante_vermelho.png");
}