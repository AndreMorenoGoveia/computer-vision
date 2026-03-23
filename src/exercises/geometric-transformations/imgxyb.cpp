//imgxyb.cpp 2024
#include "procimagem.h"
int main() {
  ImgXyb<uchar> a(3,3);
  a << 1,2,3,
       4,5,6,
       7,8,9;
  a.centro(1,1); a.backg=255;
  printf("a(-1,-1)=%d\n",a(-1,-1));
  printf("a(-1,+1)=%d\n",a(-1,+1));
  printf("minx=%d maxx=%d miny=%d maxy=%d\n",
         a.minx, a.maxx, a.miny, a.maxy);
  printf("a(-2,-1)=%d\n",a(-2,-1));
}