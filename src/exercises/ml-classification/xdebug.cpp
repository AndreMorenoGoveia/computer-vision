//xdebug.cpp
#include "procimagem.h"
int main() {
  xdebug;
  Mat_<float> a(2,2);
  xdebug;
  a << 6, 3, 9, 4;
  xdebug;
  printf("Indice=%d\n",argMax(a));  
  xdebug;
}
