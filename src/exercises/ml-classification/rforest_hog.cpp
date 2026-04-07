// rforest_hog.cpp 2026
// Linkar com OpenCV2
//
// Leitura feita do mesmo jeito que em svm.cpp:
//   MNIST mnist(...);
//   mnist.le("/home/andre/cekeikon5/tiny_dnn/data");
//
// Melhorias aplicadas para reduzir o erro sem usar SVM:
// 1) bounding box e redimensionamento para 20x20;
// 2) deskew;
// 3) HOG;
// 4) data augmentation com shifts e pequenas rotacoes;
// 5) classificacao por kNN exato, combinando HOG e pixels deskewed.
//
// Preencha depois de rodar:
// - taxa de erro observada: __.__%
// - tempo de preprocessamento: ______ s
// - tempo de treino: ______ s
// - tempo de predicao: ____ s

#include "procimagem.h"

namespace {

const int kImageSide = 20;
const double kRotations[] = {-7.0, 7.0};
const int kShifts[][2] = {
  {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

Mat_<float> shiftImage(const Mat_<float>& img, int dx, int dy) {
  Mat_<float> shifted(img.rows, img.cols, 0.0f);
  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      int sx = x - dx;
      int sy = y - dy;
      if (0 <= sx && sx < img.cols && 0 <= sy && sy < img.rows) {
        shifted(y, x) = img(sy, sx);
      }
    }
  }
  return shifted;
}

Mat_<float> rotateImage(const Mat_<float>& img, double angle) {
  Point2f center(static_cast<float>(img.cols) / 2.0f,
                 static_cast<float>(img.rows) / 2.0f);
  Mat rot = getRotationMatrix2D(center, angle, 1.0);
  Mat_<float> out;
  warpAffine(img, out, rot, img.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0.0f));
  return out;
}

Mat_<float> deskew(const Mat_<float>& img) {
  Mat momentsInput;
  img.convertTo(momentsInput, CV_32F, 255.0);
  Moments m = moments(momentsInput, false);
  if (fabs(m.mu02) < 1e-2) {
    return img.clone();
  }

  float skew = static_cast<float>(m.mu11 / m.mu02);
  Mat_<float> warp = (Mat_<float>(2, 3) << 1.0f, skew, -0.5f * img.rows * skew,
                                            0.0f, 1.0f, 0.0f);
  Mat_<float> out;
  warpAffine(img, out, warp, img.size(), WARP_INVERSE_MAP | INTER_LINEAR,
             BORDER_CONSTANT, Scalar(0.0f));
  return out;
}

vector<float> flattenAndNormalize(const Mat_<float>& img) {
  vector<float> feat;
  feat.reserve(img.rows * img.cols);
  float norm2 = 0.0f;
  for (int l = 0; l < img.rows; l++) {
    for (int c = 0; c < img.cols; c++) {
      float v = img(l, c);
      feat.push_back(v);
      norm2 += v * v;
    }
  }
  norm2 = sqrt(norm2 + 1e-12f);
  for (size_t i = 0; i < feat.size(); i++) {
    feat[i] /= norm2;
  }
  return feat;
}

vector<float> computeHog(const Mat_<float>& img) {
  static HOGDescriptor hog(
    Size(kImageSide, kImageSide),
    Size(8, 8),
    Size(4, 4),
    Size(4, 4),
    9
  );

  Mat img8u;
  img.convertTo(img8u, CV_8U, 255.0);
  vector<float> desc;
  hog.compute(img8u, desc, Size(0, 0), Size(0, 0));

  float norm2 = 0.0f;
  for (size_t i = 0; i < desc.size(); i++) {
    norm2 += desc[i] * desc[i];
  }
  norm2 = sqrt(norm2 + 1e-12f);
  for (size_t i = 0; i < desc.size(); i++) {
    desc[i] /= norm2;
  }
  return desc;
}

void appendAugmentedVersions(const Mat_<float>& base, vector< Mat_<float> >& out) {
  out.push_back(base);
  for (int s = 1; s < 5; s++) {
    out.push_back(shiftImage(base, kShifts[s][0], kShifts[s][1]));
  }
  for (int r = 0; r < 2; r++) {
    out.push_back(rotateImage(base, kRotations[r]));
  }
}

void buildTrainFeatures(MNIST& mnist, Mat_<float>& trainH, Mat_<float>& trainP, Mat_<float>& trainY) {
  vector< vector<float> > hogFeat;
  vector< vector<float> > pixFeat;
  vector<int> labels;
  hogFeat.reserve(mnist.AX.size() * 7);
  pixFeat.reserve(mnist.AX.size() * 7);
  labels.reserve(mnist.AX.size() * 7);

  for (size_t i = 0; i < mnist.AX.size(); i++) {
    Mat_<float> base;
    mnist.AX[i].convertTo(base, CV_32F, 1.0 / 255.0);

    vector< Mat_<float> > variants;
    appendAugmentedVersions(base, variants);
    for (size_t j = 0; j < variants.size(); j++) {
      Mat_<float> corrected = deskew(variants[j]);
      hogFeat.push_back(computeHog(corrected));
      pixFeat.push_back(flattenAndNormalize(corrected));
      labels.push_back(mnist.AY[i]);
    }
  }

  trainH.create(static_cast<int>(hogFeat.size()), static_cast<int>(hogFeat[0].size()));
  trainP.create(static_cast<int>(pixFeat.size()), static_cast<int>(pixFeat[0].size()));
  trainY.create(static_cast<int>(labels.size()), 1);
  for (int l = 0; l < trainH.rows; l++) {
    for (int c = 0; c < trainH.cols; c++) {
      trainH(l, c) = hogFeat[l][c];
    }
    for (int c = 0; c < trainP.cols; c++) {
      trainP(l, c) = pixFeat[l][c];
    }
    trainY(l) = static_cast<float>(labels[l]);
  }
}

void buildTestFeatures(MNIST& mnist, Mat_<float>& testH, Mat_<float>& testP) {
  vector< vector<float> > hogFeat;
  vector< vector<float> > pixFeat;
  hogFeat.reserve(mnist.QX.size());
  pixFeat.reserve(mnist.QX.size());

  for (size_t i = 0; i < mnist.QX.size(); i++) {
    Mat_<float> img;
    mnist.QX[i].convertTo(img, CV_32F, 1.0 / 255.0);
    Mat_<float> corrected = deskew(img);
    hogFeat.push_back(computeHog(corrected));
    pixFeat.push_back(flattenAndNormalize(corrected));
  }

  testH.create(static_cast<int>(hogFeat.size()), static_cast<int>(hogFeat[0].size()));
  testP.create(static_cast<int>(pixFeat.size()), static_cast<int>(pixFeat[0].size()));
  for (int l = 0; l < testH.rows; l++) {
    for (int c = 0; c < testH.cols; c++) {
      testH(l, c) = hogFeat[l][c];
    }
    for (int c = 0; c < testP.cols; c++) {
      testP(l, c) = pixFeat[l][c];
    }
  }
}

} // namespace

int main() {
  MNIST mnist(kImageSide, true, true);
  mnist.le("/home/andre/cekeikon5/tiny_dnn/data");

  Mat_<float> trainH, trainP, trainY, testH, testP;

  double t0 = tempo();
  buildTrainFeatures(mnist, trainH, trainP, trainY);
  buildTestFeatures(mnist, testH, testP);
  double t1 = tempo();

  const int kHog = 11;
  const int kPix = 5;
  CvKNearest knnHog(trainH, trainY, Mat(), false, kHog);
  CvKNearest knnPix(trainP, trainY, Mat(), false, kPix);
  double t2 = tempo();

  Mat_<float> resultH, respH(testH.rows, kHog), distH(testH.rows, kHog);
  Mat_<float> resultP, respP(testP.rows, kPix), distP(testP.rows, kPix);
  knnHog.find_nearest(testH, kHog, resultH, respH, distH);
  knnPix.find_nearest(testP, kPix, resultP, respP, distP);

  for (int l = 0; l < mnist.nq; l++) {
    double score[10] = {0};
    for (int k = 0; k < kHog; k++) {
      int label = cvRound(respH(l, k));
      score[label] += 0.80 / sqrt(distH(l, k) + 1e-9);
    }
    for (int k = 0; k < kPix; k++) {
      int label = cvRound(respP(l, k));
      score[label] += 0.20 / sqrt(distP(l, k) + 1e-9);
    }

    int best = 0;
    for (int c = 1; c < 10; c++) {
      if (score[c] > score[best]) {
        best = c;
      }
    }
    mnist.qp(l) = best;
  }
  double t3 = tempo();

  printf("Erros=%10.2f%%\n", 100.0 * mnist.contaErros() / mnist.nq);
  printf("Tempo de preprocessamento: %f\n", t1 - t0);
  printf("Tempo de treino: %f\n", t2 - t1);
  printf("Tempo de predicao: %f\n", t3 - t2);
}
