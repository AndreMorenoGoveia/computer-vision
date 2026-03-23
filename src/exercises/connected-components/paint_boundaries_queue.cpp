// paint_boundaries_queue.cpp
#include <opencv2/opencv.hpp>
#include <queue>
using namespace std;
using namespace cv;
Mat_<Vec3b> pintaAzul(Mat_<Vec3b> a, int ls, int cs)
{
    Mat_<Vec3b> b = a.clone();
    queue<int> q;
    q.push(ls);
    q.push(cs); //(1)
    while (!q.empty())
    { //(2)
        int l = q.front();
        q.pop(); //(3)
        int c = q.front();
        q.pop(); //(3)
        if (b(l, c) == Vec3b(255, 255, 255))
        {                               //(4)
            b(l, c) = Vec3b(255, 0, 0); //(5)
            q.push(l - 1);
            q.push(c); // 6-acima
            q.push(l + 1);
            q.push(c); // 6-abaixo
            q.push(l);
            q.push(c + 1); // 6-direita
            q.push(l);
            q.push(c - 1); // 6-esq
        }
    }
    return b;
}
int main()
{
    Mat_<Vec3b> a = imread("assets/mickey_reduz.bmp", 1);
    Mat_<Vec3b> b = pintaAzul(a, 159, 165);
    imwrite("results/fila.png", b);
}
