// mostra_cv.cpp
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main()
{
    Mat_<Vec3b> a = imread("assets/lenna.jpg", 1);
    imshow("janela", a);
    waitKey(0);
}
