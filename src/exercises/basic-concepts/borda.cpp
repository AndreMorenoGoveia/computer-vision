// borda.cpp
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main()
{
    Mat_<uchar> a = imread("assets/mickey_reduz.bmp", 0);
    Mat_<uchar> b(a.rows, a.cols);
    for (int l = 0; l < a.rows - 1; l++)
        for (int c = 0; c < a.cols - 1; c++)
            if (a(l, c) != a(l, c + 1) || a(l, c) != a(l + 1, c))
                b(l, c) = 0;
            else
                b(l, c) = 255;
    imwrite("results/borda.bmp", b);
}
