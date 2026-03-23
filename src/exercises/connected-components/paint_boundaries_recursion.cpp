// recursao.cpp - grad2018
#include <cekeikon.h>
void pintaAzul(Mat_<COR> &a, int l, int c)
{
    if (a(l, c) == COR(255, 255, 255))
    {
        a(l, c) = COR(255, 0, 0);
        pintaAzul(a, l - 1, c);
        pintaAzul(a, l, c - 1);
        pintaAzul(a, l + 1, c);
        pintaAzul(a, l, c + 1);
    }
}
int main()
{
    Mat_<COR> a;
    le(a, "assets/mickey_reduz.bmp");
    pintaAzul(a, 159, 165);
    imp(a, "results/recursao.png");
}
