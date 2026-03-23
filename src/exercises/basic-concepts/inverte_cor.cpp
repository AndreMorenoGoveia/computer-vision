// inverte_cor.cpp
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main()
{
    Mat_<Vec3b> a;
    a = imread("assets/lenna.jpg", 1);
    for (int l = 0; l < a.rows; l++)
        for (int c = 0; c < a.cols; c++)
        {                                  // BGR
            a(l, c)[0] = 255 - a(l, c)[0]; // blue
            a(l, c)[1] = 255 - a(l, c)[1]; // green
            a(l, c)[2] = 255 - a(l, c)[2]; // red
        }
    imwrite("results/invertec.jpg", a);
}
