//leimp.cpp
#include "procimagem.h"
int main() {
  Mat_<float> A(3,3);
  A << 1,2,3,  4,5,6,  7,8,9;
  imp(A,"results/a.txt");
  Mat_<float> B;
  le(B,"results/a.txt");
  imp(B,"/dev/tty"); //Em Windows, imp(B,"con:");
}
