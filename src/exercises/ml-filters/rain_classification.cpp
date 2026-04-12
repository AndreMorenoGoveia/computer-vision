#include "procimagem.h"

int main() {
  Mat_<Vec3b> ax=imread("assets/janei.pgm",1);
  Mat_<uchar> ay=imread("assets/janei-1.pgm",0);
  Mat_<Vec3b> qx=imread("assets/julho.pgm", 1);
  //std::cout << ax.size() << " " << ay.size() << std::endl;
  if (ax.size()!=ay.size()) erro("Erro dimensao");
  Mat_<uchar> qp(qx.rows,qx.cols);

  //Cria as estruturas de dados para alimentar OpenCV
  Mat_<float> features(ax.rows*ax.cols,9);
  Mat_<int> saidas(ax.rows*ax.cols,1);
  int i=0;
  for (int l=0; l<ax.rows; l++)
    for (int c=0; c<ax.cols; c++) {
      int k=0;
      for (int dl=-1; dl<=1; dl++)
        for (int dc=-1; dc<=1; dc++) {
          int ll=max(0,min(ax.rows-1,l+dl));
          int cc=max(0,min(ax.cols-1,c+dc));
          features(i,k++)=ax(ll,cc)[0]/255.0;
        }
      saidas(i)=ay(l,c);
      i=i+1;
    }
  flann::Index ind(features,flann::KDTreeIndexParams(4));
  // Aqui, as 4 arvores estao criadas

  Mat_<float> query(1,9);
  vector<int> indices(2);
  vector<float> dists(2);
  for (int l=0; l<qp.rows; l++)
    for (int c=0; c<qp.cols; c++) {
      int k=0;
      for (int dl=-1; dl<=1; dl++)
        for (int dc=-1; dc<=1; dc++) {
          int ll=max(0,min(qx.rows-1,l+dl));
          int cc=max(0,min(qx.cols-1,c+dc));
          query(0,k++)=qx(ll,cc)[0]/255.0;
        }
      // Zero indica sem backtracking
      ind.knnSearch(query,indices,dists,1,flann::SearchParams(0));
      qp(l,c)=saidas(indices[0]);
    }
  //imwrite("results/f1-rain-flann.png",qp);

  Mat_<Vec3b> qpred=qx.clone();
  for (int l=0; l<qp.rows; l++)
    for (int c=0; c<qp.cols; c++) {
      int contagemPreto=0;
      for (int dl=-1; dl<=1; dl++)
        for (int dc=-1; dc<=1; dc++) {
          int ll=max(0,min(qp.rows-1,l+dl));
          int cc=max(0,min(qp.cols-1,c+dc));
          if (qp(ll,cc)==0) contagemPreto++;
        }
      if (contagemPreto>=8)
        qpred(l,c)[2]=255;
    }
  imwrite("results/f1-rain-flann-red-final.png",qpred);
}
