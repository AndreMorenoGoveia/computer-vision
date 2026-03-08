# Computer Vision

Repositório pessoal dedicado a estudos e implementações em visao computacional, cobrindo processamento de imagens com C++ e OpenCV, alem de fluxos de deep learning com Python e Keras.

## Objetivos

- Consolidar experimentos e implementacoes de processamento de imagens.
- Organizar exercicios praticos, projetos maiores e iteracoes de modelos.
- Manter uma base unica para codigo, dados, resultados e documentacao.

## Stack principal

- `C++` com `OpenCV` para processamento de imagens, transformacoes, filtros e visao classica.
- `Python` com `Keras` para redes neurais, classificacao e pipelines de deep learning.

## Estrutura do projeto

```text
computer-vision/
├── artifacts/
│   ├── models/                 # modelos treinados, checkpoints e pesos exportados
│   └── results/                # saidas geradas, logs e imagens resultantes
├── data/
│   ├── datasets/               # datasets tabulares ou estruturados para treinamento
│   └── images/
│       ├── processed/          # imagens tratadas e derivadas
│       └── raw/                # imagens originais recebidas ou coletadas
├── docs/                       # anotacoes, referencias e relatorios
├── include/
│   └── cpp/
│       └── opencv/             # headers compartilhados do codigo C++
├── notebooks/                  # exploracoes e prototipos em Jupyter
├── scripts/                    # automacoes auxiliares
├── src/
│   ├── cpp/
│   │   └── opencv/
│   │       ├── common/         # utilitarios reutilizaveis
│   │       ├── exercises/
│   │       │   ├── home/       # exercicios desenvolvidos de forma independente
│   │       │   └── studio/     # exercicios curtos e experimentos guiados
│   │       └── projects/
│   │           └── eps/        # entregas maiores e projetos estruturados
│   └── python/
│       └── keras/
│           ├── common/         # funcoes compartilhadas, preprocessamento e helpers
│           ├── exercises/
│           │   ├── home/       # exercicios desenvolvidos de forma independente
│           │   └── studio/     # exercicios curtos e experimentos guiados
│           └── projects/
│               └── eps/        # entregas maiores e projetos estruturados
└── tests/
    ├── cpp/                    # testes para modulos em C++
    └── python/                 # testes para modulos em Python
```

## Convencao de organizacao

- Coloque novos codigos em `src/`, separados por linguagem e contexto.
- Mantenha imagens originais em `data/images/raw/` e gere derivados em `data/images/processed/`.
- Salve pesos, checkpoints e saidas em `artifacts/` para evitar misturar resultados com codigo-fonte.
- Use `common/` para componentes reutilizaveis e evite duplicacao entre exercicios e projetos.
- Registre observacoes tecnicas e decisoes em `docs/` quando um experimento crescer.

## Como evoluir este repositorio

1. Adicionar um sistema de build para o codigo C++ em `src/cpp/` com `CMake`.
2. Criar um ambiente Python com `venv` ou `conda` para os projetos em `src/python/`.
3. Documentar dependencias, datasets e passos de execucao por projeto dentro de `docs/` ou em READMEs locais.

## Licenca

Este projeto esta distribuido sob os termos definidos em `LICENSE`.
