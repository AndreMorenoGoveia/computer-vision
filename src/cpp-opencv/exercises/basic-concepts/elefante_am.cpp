// elefante_am.cpp 2024
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;
int distancia(Vec3b a, Vec3b b)
{
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
}
int main()
{
    Mat_<Vec3b> a = imread("assets/elefante.jpg", 1);
    Mat_<Vec3b> b;
    Vec3b amarelo(20, 200, 200);
    
    b = a.clone();
    for (int l = 0; l < a.rows; l++)
        for (int c = 0; c < a.cols; c++)
            if (distancia(amarelo, a(l, c)) < 100)
                b(l, c) = Vec3b(0, 0, 255);
    imwrite("results/elefante_am.png", b);
}
