//flann.cpp 2024
//segmentacao de feijao usando FlaNN
//Pode usar OpenCV v2, v3 ou v4
#include "procimagem.h"

int main() {
  Mat_<Vec3b> ax=imread("assets/ax.png",1);
  Mat_<uchar> ay=imread("assets/ay.png",0);
  Mat_<Vec3b> qx=imread("assets/f1.jpg",1);
  if (ax.size()!=ay.size()) erro("Erro dimensao");
  Mat_<uchar> qp(qx.rows,qx.cols);

  //Cria as estruturas de dados para alimentar OpenCV
  Mat_<float> features(ax.rows*ax.cols,3);
  Mat_<int> saidas(ax.rows*ax.cols,1);
  int i=0;
  for (int l=0; l<ax.rows; l++)
    for (int c=0; c<ax.cols; c++) {
      features(i,0)=ax(l,c)[0]/255.0;
      features(i,1)=ax(l,c)[1]/255.0;
      features(i,2)=ax(l,c)[2]/255.0;
      saidas(i)=ay(l,c);
      i=i+1;
    }
  flann::Index ind(features,flann::KDTreeIndexParams(4));
  // Aqui, as 4 arvores estao criadas

  Mat_<float> query(1,3);
  vector<int> indices(1);
  vector<float> dists(1);
  for (int l=0; l<qp.rows; l++)
    for (int c=0; c<qp.cols; c++) {
      query(0)=qx(l,c)[0]/255.0;
      query(1)=qx(l,c)[1]/255.0;
      query(2)=qx(l,c)[2]/255.0;
      // Zero indica sem backtracking
      ind.knnSearch(query,indices,dists,1,flann::SearchParams(0));
      qp(l,c)=saidas(indices[0]);
    }
  imwrite("results/f1-flann.png",qp);
}
