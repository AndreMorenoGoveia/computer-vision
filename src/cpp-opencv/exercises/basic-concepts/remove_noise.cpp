// remove_noise.cpp
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

bool colorChainIsLongEnough(Mat_<uchar> a, int l, int c,
     uint8_t currentLength, uchar currentColor);

int main()
{
    Mat_<uchar> a = imread("assets/mickeyr.bmp", 0);

    for (int l = 0; l < a.rows; l++){
        for (int c = 0; c < a.cols; c++){
            
            uint8_t differentNeighbors = 0; 

            if(l == 0 || a(l - 1, c) != a(l, c)){
                differentNeighbors++;
            }
            if(l == a.rows - 1 || a(l + 1, c) != a(l, c)){
                differentNeighbors++;
            }
            if(c == 0 || a(l, c - 1) != a(l, c)){
                differentNeighbors++;
            }
            if(c == a.cols - 1 || a(l, c + 1) != a(l, c)){
                differentNeighbors++;
            }

            if(differentNeighbors < 3)
                continue;
            
            if (a(l, c) == 0)
                a(l, c) = 255;
            else
                a(l, c) = 0;
        }
    }

    for (int l = 0; l < a.rows; l++){
        for (int c = 0; c < a.cols; c++){
            
            if(colorChainIsLongEnough(a, l, c, 1, a(l,c)))
                continue;
            
            if (a(l, c) == 0)
                a(l, c) = 255;
            else
                a(l, c) = 0;
        }
    }

    imwrite("results/eliminaruibr.bmp", a);
}


bool colorChainIsLongEnough(Mat_<uchar> a, int l, int c,
     uint8_t currentLength, uchar currentColor){
    if(l < 0 || c < 0 || l >= a.rows || c >= a.cols)
        return false;
    
    if(a(l,c) != currentColor)
        return false;

    if(currentLength >= 5)
        return true;

    return 1 == 0
    || colorChainIsLongEnough(a, l, c + 1, currentLength+1, a(l,c))
    || colorChainIsLongEnough(a, l, c - 1, currentLength+1, a(l,c))
    || colorChainIsLongEnough(a, l + 1, c, currentLength+1, a(l,c))
    || colorChainIsLongEnough(a, l - 1, c, currentLength+1, a(l,c));
    
    
}