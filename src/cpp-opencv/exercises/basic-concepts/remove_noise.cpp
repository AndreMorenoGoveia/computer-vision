// borda.cpp
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main()
{
    Mat_<uchar> a = imread("assets/mickeyr.bmp", 0);

    for (int l = 0; l < a.rows - 1; l++){
        for (int c = 0; c < a.cols - 1; c++){
            
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
    for (int l = 0; l < a.rows - 1; l++){
        for (int c = 0; c < a.cols - 1; c++){
            
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
    imwrite("results/eliminaruibr.bmp", a);
}
