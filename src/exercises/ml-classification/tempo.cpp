//tempo.cpp
#include "procimagem.h"
int main() {
  double t1=tempo();
  this_thread::sleep_for(1s); //Troque por algum processamento
  double t2=tempo();
  printf("Tempo decorrido em segundos=%f\n",t2-t1);
}