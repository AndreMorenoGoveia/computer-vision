# EP - ViT vs CNN no Oxford Flowers 102

PSI5790, primeiro periodo de 2026. Compara EfficientNetB0 e DeiT-Tiny no
Oxford Flowers 102 sob oclusao central e distribuida, e busca um modelo
de maior acuracia (Objetivo 2).

## Estrutura

```
ep-vit-cnn/
├── common.py        # carregamento do dataset, oclusoes, helpers
├── train_cnn.py     # Tarefa 1 - EfficientNetB0
├── train_vit.py     # Tarefa 1 - DeiT-Tiny
├── train_best.py    # Objetivo 2 - ViT-Base
├── evaluate.py      # Tarefas 2 e 3 (oclusoes) + previews
├── relatorio.md     # Relatorio final em Markdown
└── results/         # graficos, JSONs e imagens geradas
```

Modelos treinados sao salvos em `artifacts/models/ep-vit-cnn/`.

## Dependencias

Recomendado Google Colab com GPU. Instale:

```bash
pip install tensorflow tensorflow-datasets keras-hub matplotlib
```

## Execucao

```bash
# A partir da pasta do EP
cd src/projects/ep-vit-cnn

python train_cnn.py     # Tarefa 1 - CNN
python train_vit.py     # Tarefa 1 - ViT
python train_best.py    # Objetivo 2

python evaluate.py      # Tarefas 2, 3 e Objetivo 2 sob oclusoes
```

`evaluate.py` carrega os checkpoints salvos e produz:

- `results/train_preview.png`, `results/test_clean_preview.png`,
  `results/test_central_preview.png`, `results/test_distributed_preview.png`
- `results/final_metrics.json` com a tabela final
- Logs com as acuracias para cada modelo e cada cenario

## Observacoes

- O ambiente local nao tem GPU. Treinar EfficientNetB0 + DeiT-Tiny + ViT-Base
  em CPU e inviavel. Os scripts foram escritos para rodar em Colab/maquina
  com GPU CUDA.
- A oclusao distribuida usa `seed=42` (em `common.apply_distributed_occlusion`)
  para que os resultados sejam reprodutiveis entre execucoes.
- BatchNormalization da EfficientNetB0 e mantida congelada durante todo o
  fine-tuning, conforme sugerido no enunciado.
- Uso de IA: este EP foi estruturado com auxilio do Claude Code para gerar
  esqueletos dos scripts; a inspecao do codigo, decisoes de hiperparametros
  e analise dos resultados sao do autor.
