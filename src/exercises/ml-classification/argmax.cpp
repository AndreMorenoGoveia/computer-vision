//argmax.cpp
#include "procimagem.h"
int main() {
  Mat_<float> a(2,2);
  a << 6, 3, 9, 4;
  printf("Indice do maior elemento=%d\n",argMax(a));  
}
