# MNIST Classico Com SVM + HOG

Este experimento usa o arquivo [svm_sklearn_hog.py](/home/andre/repos/computer-vision/src/exercises/ml-classification/svm_sklearn_hog.py) para classificar MNIST com aprendizado de maquina classico, sem CNN.

Durante os testes deste projeto, foi obtida taxa de erro de `0,78%` com a linha de trabalho em Python. Depois disso, foi tentado um ensemble mais complexo, mas ele piorou para `1,47%`. Por isso, a versao atual voltou para a base mais estavel:

- `bounding box`
- `deskew`
- `HOG`
- `data augmentation`
- `LinearSVC`

Ou seja: em vez de acumular complexidade, a implementacao ficou concentrada no que realmente entregou melhor resultado.

## Objetivo

Baixar ao maximo a taxa de erro em MNIST usando apenas tecnicas classicas de visao computacional e aprendizado de maquina.

## Como o dado e transformado

O MNIST original tem imagens `28x28`, com digito branco sobre fundo preto. O script transforma esse dado em varias etapas.

### 1. Leitura

O programa tenta ler primeiro os arquivos IDX em:

- `/home/andre/cekeikon5/tiny_dnn/data`

Se nao encontrar, ele cai para o CSV em:

- [assets/MNIST_CSV](/home/andre/repos/computer-vision/src/exercises/ml-classification/assets/MNIST_CSV)

### 2. Inversao de cores

As cores sao invertidas para ficar no mesmo estilo do `procimagem.h`:

- antes: digito claro em fundo escuro
- depois: digito escuro em fundo claro

Isso ajuda a manter o pipeline coerente com os exemplos da disciplina.

### 3. Bounding Box

Depois da inversao, o programa remove as bordas claras externas e fica so com a regiao do digito.

Intuicao:

- o fundo nao ajuda a classificar
- a bounding box reduz a dimensionalidade
- a imagem passa a concentrar mais informacao util

Depois disso, a imagem e redimensionada para `20x20`.

### 4. Deskew

Mesmo depois da bounding box, muitos digitos ainda ficam levemente inclinados. O `deskew` corrige isso usando momentos da imagem.

O que ele faz:

- calcula a massa do digito
- mede o cisalhamento pela relacao `mu11 / mu02`
- aplica uma transformacao afim para alinhar melhor o traco

Na pratica, isso ajuda bastante a reduzir variacoes desnecessarias entre exemplos da mesma classe.

### 5. HOG

Com a imagem alinhada, o script extrai `HOG`:

- imagem `20x20`
- celulas `4x4`
- `9` bins de orientacao
- normalizacao por blocos `2x2`

Por que HOG ajuda:

- capta contorno e direcao dos gradientes
- fica mais robusto do que pixels crus
- usa melhor a relacao de vizinhanca entre pixels

## Data Augmentation no treino

Para cada amostra de treino, o script gera:

- original
- deslocamento para norte
- deslocamento para sul
- deslocamento para leste
- deslocamento para oeste
- rotacao `-7`
- rotacao `+7`

Isso aumenta a variedade do conjunto de treino e deixa o modelo menos sensivel a pequenas mudancas geométricas.

## Classificador usado

O modelo final e:

- `StandardScaler`
- `LinearSVC`

Essa escolha foi mantida porque foi a base mais estavel do experimento. O ensemble adicional com uma segunda visao `10x10 sem bounding box` piorou o erro e foi removido.

## Por que essa versao ficou melhor

Os principais fatores que ajudam o resultado final sao:

- `bounding box`: remove informacao irrelevante
- `20x20`: reduz dimensao sem destruir a forma
- `deskew`: diminui variacao angular desnecessaria
- `HOG`: destaca a geometria do digito
- `augmentation`: ensina invariancia a pequenas mudancas
- `LinearSVC`: aprende uma fronteira forte em cima dos atributos HOG

## Como rodar

Na raiz do repositorio:

```bash
python3 src/exercises/ml-classification/svm_sklearn_hog.py
```

Saida esperada:

- taxa de erro
- tempo de preprocessamento
- tempo de treino
- tempo de predicao

## Caches gerados

O script salva o cache das features de treino em:

- [results](/home/andre/repos/computer-vision/src/exercises/ml-classification/results)

Esses arquivos `.npz` podem ficar muito grandes, por isso estao ignorados no git.

## Como tentar ir alem de 0,78%

As proximas tentativas mais promissoras sao:

- busca fina do parametro `C` do `LinearSVC`
- testar `24x24` em vez de `20x20`
- adicionar test-time augmentation novamente, mas so se ela ajudar empiricamente
- testar concatenacao de `HOG + pixels deskewed`
- usar uma segunda visao novamente, mas so se ela provar empiricamente que ajuda

## Arquivos principais

- [svm_sklearn_hog.py](/home/andre/repos/computer-vision/src/exercises/ml-classification/svm_sklearn_hog.py): pipeline principal em Python
- [rforest_hog.cpp](/home/andre/repos/computer-vision/src/exercises/ml-classification/rforest_hog.cpp): experimento classico em C++
- [svm.cpp](/home/andre/repos/computer-vision/src/exercises/ml-classification/svm.cpp): baseline da aula
