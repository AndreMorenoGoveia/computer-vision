// pedestre_carro_hog_svm.cpp 2026
// Exercício extra: classificar imagem como pedestre ou carro
// Dataset esperado em assets/pedestre e assets/carro
// Compilar pelo Makefile da pasta

#include "procimagem.h"
#include <cctype>
#include <cstdio>

namespace {

const int kPedestre = 1;
const int kCarro = -1;
const int kTrainPedestres = 399;
const int kTrainCarrosPorSerie = 199;

struct Sample {
  string path;
  int label;
};

struct EvalStats {
  int correct;
  int pedestreAsPedestre;
  int pedestreAsCarro;
  int carroAsCarro;
  int carroAsPedestre;
  string firstErrorPath;
  int firstExpected;
  int firstPredicted;
};

bool fileExists(const string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == NULL) return false;
  fclose(f);
  return true;
}

string resolveAssetsRoot() {
  const string candidates[] = {
    "assets",
    "src/exercises/ml-classification/assets"
  };
  for (int i = 0; i < 2; i++) {
    if (fileExists(candidates[i] + "/pedestre/per00001.ppm")) return candidates[i];
  }
  erro("Nao encontrei a pasta assets do exercicio");
  return "";
}

int extractId(const string& path) {
  size_t slash = path.find_last_of("/\\");
  string name = (slash == string::npos ? path : path.substr(slash + 1));
  size_t dot = name.find_last_of('.');
  string stem = (dot == string::npos ? name : name.substr(0, dot));

  int ini = static_cast<int>(stem.size());
  while (ini > 0 && isdigit(static_cast<unsigned char>(stem[ini - 1]))) ini--;
  if (ini == static_cast<int>(stem.size())) erro("Nome de arquivo sem numero: " + path);
  return atoi(stem.substr(ini).c_str());
}

void addSplitSamples(const string& pattern,
                     int trainUntil,
                     int label,
                     vector<Sample>& train,
                     vector<Sample>& test) {
  vector<string> files;
  glob(pattern, files, false);
  if (files.empty()) erro("Nenhum arquivo encontrado para " + pattern);

  for (size_t i = 0; i < files.size(); i++) {
    Sample sample{files[i], label};
    if (extractId(files[i]) <= trainUntil) train.push_back(sample);
    else test.push_back(sample);
  }
}

vector<float> computeHog(const Mat& bgr) {
  static HOGDescriptor hog(
    Size(64, 128),
    Size(32, 32),
    Size(16, 16),
    Size(16, 16),
    9,
    1,
    -1,
    0,
    0.2,
    false,
    HOGDescriptor::DEFAULT_NLEVELS
  );

  Mat gray;
  if (bgr.channels() == 3) cvtColor(bgr, gray, CV_BGR2GRAY);
  else gray = bgr;

  if (gray.cols != 64 || gray.rows != 128) {
    resize(gray, gray, Size(64, 128), 0, 0, INTER_LINEAR);
  }

  vector<float> desc;
  hog.compute(gray, desc, Size(8, 8), Size(0, 0));
  return desc;
}

void buildDataset(const vector<Sample>& samples, Mat_<float>& data, Mat_<float>& labels) {
  vector< vector<float> > feats;
  feats.reserve(samples.size());

  for (size_t i = 0; i < samples.size(); i++) {
    Mat img = imread(samples[i].path, 1);
    if (img.empty()) erro("Falha lendo imagem " + samples[i].path);
    feats.push_back(computeHog(img));
  }

  if (feats.empty()) erro("Dataset vazio");

  data.create(static_cast<int>(feats.size()), static_cast<int>(feats[0].size()));
  labels.create(static_cast<int>(feats.size()), 1);

  for (int l = 0; l < data.rows; l++) {
    for (int c = 0; c < data.cols; c++) data(l, c) = feats[l][c];
    labels(l) = static_cast<float>(samples[l].label);
  }
}

double evaluate(const CvSVM& svm, const vector<Sample>& test, EvalStats& stats) {
  stats.correct = 0;
  stats.pedestreAsPedestre = 0;
  stats.pedestreAsCarro = 0;
  stats.carroAsCarro = 0;
  stats.carroAsPedestre = 0;
  stats.firstErrorPath = "";
  stats.firstExpected = 0;
  stats.firstPredicted = 0;

  for (size_t i = 0; i < test.size(); i++) {
    Mat img = imread(test[i].path, 1);
    if (img.empty()) erro("Falha lendo imagem " + test[i].path);

    vector<float> desc = computeHog(img);
    Mat_<float> sample(1, static_cast<int>(desc.size()));
    for (int c = 0; c < sample.cols; c++) sample(0, c) = desc[c];

    int pred = cvRound(svm.predict(sample));
    if (pred == test[i].label) {
      stats.correct++;
    } else if (stats.firstErrorPath.empty()) {
      stats.firstErrorPath = test[i].path;
      stats.firstExpected = test[i].label;
      stats.firstPredicted = pred;
    }

    if (test[i].label == kPedestre) {
      if (pred == kPedestre) stats.pedestreAsPedestre++;
      else stats.pedestreAsCarro++;
    } else {
      if (pred == kCarro) stats.carroAsCarro++;
      else stats.carroAsPedestre++;
    }
  }

  return 100.0 * stats.correct / test.size();
}

} // namespace

int main() {
  const string assetsRoot = resolveAssetsRoot();

  vector<Sample> train, test;
  addSplitSamples(assetsRoot + "/pedestre/per*.ppm", kTrainPedestres, kPedestre, train, test);
  addSplitSamples(assetsRoot + "/carro/carl*.ppm", kTrainCarrosPorSerie, kCarro, train, test);
  addSplitSamples(assetsRoot + "/carro/carr*.ppm", kTrainCarrosPorSerie, kCarro, train, test);

  Mat_<float> trainX, trainY;
  double t0 = tempo();
  buildDataset(train, trainX, trainY);
  double t1 = tempo();

  CvSVMParams params;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.C = 1.0;
  params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

  CvSVM svm;
  svm.train(trainX, trainY, Mat(), Mat(), params);
  double t2 = tempo();

  EvalStats stats;
  double accuracy = evaluate(svm, test, stats);
  double t3 = tempo();

  printf("Treino: %lu imagens\n", static_cast<unsigned long>(train.size()));
  printf("Teste:  %lu imagens\n", static_cast<unsigned long>(test.size()));
  printf("Acertos=%d/%lu\n", stats.correct, static_cast<unsigned long>(test.size()));
  printf("Taxa de acerto=%10.2f%%\n", accuracy);
  printf("Taxa de erro=%12.2f%%\n", 100.0 - accuracy);
  printf("Matriz de confusao (real x predito):\n");
  printf("  pedestre -> pedestre: %d\n", stats.pedestreAsPedestre);
  printf("  pedestre -> carro:    %d\n", stats.pedestreAsCarro);
  printf("  carro    -> carro:    %d\n", stats.carroAsCarro);
  printf("  carro    -> pedestre: %d\n", stats.carroAsPedestre);
  if (!stats.firstErrorPath.empty()) {
    printf("Primeiro erro: %s\n", stats.firstErrorPath.c_str());
    printf("  esperado=%d predito=%d\n", stats.firstExpected, stats.firstPredicted);
  }
  printf("Tempo de extracao (treino): %f\n", t1 - t0);
  printf("Tempo de treino: %f\n", t2 - t1);
  printf("Tempo de teste: %f\n", t3 - t2);
}
