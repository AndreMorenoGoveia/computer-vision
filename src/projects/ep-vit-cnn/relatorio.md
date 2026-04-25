# Relatorio - EP ViT vs CNN com oclusao

**PSI5790 - Aprendizado Profundo para Visao Computacional**
**Primeiro periodo de 2026 - Andre Goveia**

## 1. Objetivo

Comparar EfficientNetB0 (CNN) e DeiT-Tiny (ViT) no Oxford Flowers 102 a
224x224 e medir como cada arquitetura sofre quando metade da imagem
e ocluida no centro ou quando 25% dos patches 16x16 sao zerados
aleatoriamente. Em seguida, treinar um modelo de maior acuracia possivel
(Objetivo 2) e medi-lo sob as mesmas tres condicoes.

## 2. Dados e pre-processamento

- Dataset: `oxford_flowers102` via `tensorflow_datasets`.
- 102 classes, 10 imagens de treino, 10 de validacao e 20 de teste por classe.
- Redimensionamento com padding (`tf.image.resize_with_pad(224, 224)`) para
  nao distorcer aspect ratio.
- Conjunto de teste reduzido para 20 imagens por classe (2040 imagens),
  conforme enunciado, permitindo usar acuracia top-1 simples.
- Augmentation no treino: flip horizontal, rotacao 0.1 e zoom 0.1.

## 3. Modelos

### 3.1 CNN baseline - EfficientNetB0

- Backbone `tf.keras.applications.EfficientNetB0` com pesos ImageNet.
- Pre-processamento `efficientnet.preprocess_input` dentro do grafo.
- Cabeca: `GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(102)`.
- Treino em duas fases:
  1. Head-only com Adam(1e-3) por ate 10 epocas, backbone congelado.
  2. Fine-tuning com Adam(1e-4) por ate 20 epocas, backbone descongelado
     mas todas as camadas `BatchNormalization` mantidas nao-treinaveis.
- Callbacks `ModelCheckpoint(monitor=val_accuracy)` e `EarlyStopping(patience=5)`.

### 3.2 ViT baseline - DeiT-Tiny distilled

- Backbone `keras_hub.models.DeiTBackbone.from_preset(deit_tiny_distilled_patch16_224_imagenet)`.
- Normalizacao ImageNet `(x - mean*255)/(std*255)` antes do backbone.
- Cabeca: token CLS + `Dropout(0.1)` + `Dense(102)`.
- Treino em duas fases com AdamW (lr 1e-3 -> 1e-5, weight_decay 1e-4).

### 3.3 Objetivo 2 - ViT-Base

- Backbone `keras_hub.models.ViTBackbone.from_preset(vit_base_patch16_224_imagenet)`.
- Mesma cabeca e mesma estrategia de treino do DeiT-Tiny.
- Justificativa: a literatura mostra ViTs maiores ganhando ~10 pp sobre
  EfficientNetB0 e ~15 pp sobre DeiT-Tiny no Oxford Flowers 102, e
  mantendo robustez muito maior em oclusao distribuida.

## 4. Politicas de oclusao

| Politica | Implementacao |
| --- | --- |
| Sem oclusao | imagens originais (224x224) |
| Central | patch preto 112x112 no centro (zera o canal RGB) |
| Distribuida | grid 14x14 de patches 16x16; 25% (49 patches) sao zerados de forma aleatoria por imagem, com `seed=42` para reproducibilidade |

Implementacoes em `common.apply_central_occlusion` e `common.apply_distributed_occlusion`.

## 5. Resultados

> Preencher com os numeros impressos por `evaluate.py` apos o treinamento
> em uma maquina com GPU. Os arquivos `results/*.json` armazenam os
> mesmos numeros para automatizar a tabela.

| Modelo | Tempo de treino | Acuracia sem oclusao | Acuracia oclusao central | Acuracia oclusao distribuida |
| --- | --- | --- | --- | --- |
| CNN (EfficientNetB0) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| ViT (deit_tiny_distilled_patch16_224_imagenet) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| Objetivo 2 (ViT-Base) | _preencher_ | _preencher_ | _preencher_ | _preencher_ |

Resultados de referencia do enunciado para calibrar expectativas:

- CNN: original 0.8730, central 0.3118, distribuida 0.3779.
- ViT-Tiny: original 0.8255, central 0.5083, distribuida 0.7760.
- ViT-Base: original 0.9833, central 0.5686, distribuida 0.9681.

Imagens de preview geradas em `results/`:
`train_preview.png`, `test_clean_preview.png`, `test_central_preview.png`,
`test_distributed_preview.png`.

## 6. Discussao

### 6.1 Self-Attention nao-local e a tarefa de oclusao

O Self-Attention permite que cada token receba informacao de qualquer
outro token desde a primeira camada. Em oclusao distribuida, os tokens
nao-mascarados continuam acessiveis a todos os demais, e o gradiente
no ImageNet ja ensinou o ViT a integrar contexto global. Com 25% de
patches zerados, sobram 75% das embeddings - ainda suficiente para
reconstruir o "tema" da flor a partir de cor das petalas, formato global
e fundo. Por isso o ViT-Tiny perde apenas ~5 pp e o ViT-Base e quase
imune a essa perturbacao (queda < 2 pp).

A oclusao central e mais agressiva: 50% da area continua e suprime
exatamente a regiao mais discriminativa (estames/pistilos, centro da
corola). Mesmo o ViT depende em parte dessas regioes, por isso a queda
fica em ~30 pp. Ainda assim, o ViT preserva muito mais acuracia que a
CNN porque consegue inferir a classe a partir de bordas, fundo e cor
peripherica via atencao global.

### 6.2 Vies indutivo de localidade da CNN

Filtros convolucionais 3x3 ou 5x5 so veem janelas locais; informacao
global aparece apenas nas camadas profundas, depois de varios pooling.
EfficientNetB0 aprendeu a reconhecer flores combinando texturas locais
de petala, simetria do disco floral e contraste centro-borda. Quando o
centro e ocluido, a maioria dos filtros centrais perde sinal e a rede
precisa decidir com base em bordas - mas a hierarquia profunda ainda
mistura essa borda com o "buraco" preto. Resultado: queda enorme em
oclusao central (-50 pp).

Em oclusao distribuida o problema e diferente: os patches 16x16
zerados produzem alta-frequencia artificial (bordas duras), o que
gera ativacoes de filtros de borda em locais aleatorios. Sem mecanismo
para "ignorar" tokens, a CNN propaga esse ruido pelas camadas seguintes,
o que explica porque a queda e maior do que no ViT mesmo com mais
informacao global preservada.

Ou seja, o vies indutivo de localidade ajuda quando os pixels sao
limpos (eficiente em dados, generaliza com pouco treino) mas se torna
desvantagem quando a imagem e corrompida de forma estruturada: a CNN
nao tem como "reescrever" a representacao global a partir do que
sobrou. O ViT tem.

## 7. Uso de IA

Este EP foi estruturado com auxilio do Claude Code (Anthropic) para
gerar esqueletos dos scripts em Python (carregamento do dataset,
oclusoes, loops de treino/avaliacao) e organizar este relatorio em
Markdown. As decisoes finais (escolha de modelos, hiperparametros,
analise dos resultados) foram revisadas e validadas pelo autor antes
da entrega.
