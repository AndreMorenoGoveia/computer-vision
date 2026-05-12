# Alterações introduzidas em relação ao programa original (tr02c.py):
#
# 1. DATA AUGMENTATION: RandomRotation (±18°), RandomTranslation (±10%),
#    RandomZoom (±10%). Amplia a variabilidade dos dados de treino sem coletar
#    novas amostras, reduzindo overfitting.
#
# 2. WARMUP + COSINE DECAY: as primeiras 5 épocas aumentam o LR linearmente
#    até o valor base; depois, decai em cosine até zero. Evita instabilidade
#    inicial e garante convergência suave no final.
#
# 3. LABEL SMOOTHING (ε=0.1): suaviza os alvos de 1.0 para 0.9, penalizando
#    predições excessivamente confiantes e melhorando generalização.
#
# 4. STOCHASTIC DEPTH (DropPath): zera aleatoriamente blocos inteiros do
#    transformer durante o treino (taxa cresce linearmente de 0 a 10%).
#    Regularização específica para redes profundas.
#
# 5. MAIS CAMADAS TRANSFORMER: 6 blocos (original: 4). Aumenta a capacidade
#    do modelo de capturar relações complexas entre patches.
#
# 6. EARLY STOPPING com restauração dos melhores pesos (patience=20).

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
NUM_CLASSES       = 10
INPUT_SHAPE       = (28, 28, 1)
LEARNING_RATE     = 0.001
WEIGHT_DECAY      = 0.0001
BATCH_SIZE        = 256
NUM_EPOCHS        = 120
PATCH_SIZE        = 7
IMAGE_SIZE        = 28
NUM_PATCHES       = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM    = 64
NUM_HEADS         = 4
TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
TRANSFORMER_LAYERS = 6
MLP_HEAD_UNITS    = [2048, 1024]
STOCH_DEPTH_MAX   = 0.10
LABEL_SMOOTHING   = 0.10
WARMUP_EPOCHS     = 5

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
x_test  = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

y_train_oh = tf.one_hot(y_train, NUM_CLASSES)
y_test_oh  = tf.one_hot(y_test,  NUM_CLASSES)

# ---------------------------------------------------------------------------
# Alteração 1: Data Augmentation
# ---------------------------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomRotation(factor=0.05),                          # ±18°
    layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
    layers.RandomZoom(height_factor=(-0.10, 0.10)),
], name="data_augmentation")

# ---------------------------------------------------------------------------
# Alteração 2: Warmup + Cosine Decay
# ---------------------------------------------------------------------------
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr      = base_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        progress  = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr":      self.base_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

# ---------------------------------------------------------------------------
# Alteração 4: Stochastic Depth
# ---------------------------------------------------------------------------
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob    = 1.0 - self.drop_prob
        shape        = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_mask  = tf.cast(tf.random.uniform(shape) < keep_prob, x.dtype)
        return (x / keep_prob) * random_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg

# ---------------------------------------------------------------------------
# Patches e PatchEncoder (iguais ao original)
# ---------------------------------------------------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches       = num_patches
        self.projection        = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def create_vit_classifier():
    inputs = layers.Input(shape=INPUT_SHAPE)

    x = data_augmentation(inputs)

    patches         = Patches(PATCH_SIZE)(x)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    dpr = np.linspace(0, STOCH_DEPTH_MAX, TRANSFORMER_LAYERS)

    for i in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attn_out = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        attn_out = StochasticDepth(drop_prob=float(dpr[i]))(attn_out)
        x2 = layers.Add()([attn_out, encoded_patches])

        # --- MLP ---
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(TRANSFORMER_UNITS[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(TRANSFORMER_UNITS[1], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = StochasticDepth(drop_prob=float(dpr[i]))(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Cabeça de classificação
    rep = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    rep = layers.Flatten()(rep)
    rep = layers.Dropout(0.5)(rep)
    for units in MLP_HEAD_UNITS:
        rep = layers.Dense(units, activation=tf.nn.gelu)(rep)
        rep = layers.Dropout(0.5)(rep)

    logits = layers.Dense(NUM_CLASSES)(rep)
    return keras.Model(inputs=inputs, outputs=logits)


model = create_vit_classifier()
model.summary()

# ---------------------------------------------------------------------------
# Compilação com LR schedule e label smoothing
# ---------------------------------------------------------------------------
steps_per_epoch = int(len(x_train) * 0.9) // BATCH_SIZE
total_steps     = steps_per_epoch * NUM_EPOCHS
warmup_steps    = steps_per_epoch * WARMUP_EPOCHS

lr_schedule = WarmupCosineDecay(
    base_lr=LEARNING_RATE,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=LABEL_SMOOTHING   # Alteração 3
    ),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20, restore_best_weights=True
    ),
]

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
print("Iniciando treinamento melhorado do Vision Transformer no MNIST...")
history = model.fit(
    x=x_train,
    y=y_train_oh,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_split=0.1,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# Curvas
# ---------------------------------------------------------------------------
output_dir = os.path.dirname(os.path.abspath(__file__))

plt.figure()
plt.plot(history.history["accuracy"],     label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, "vit_mnist_improved_acc.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"],     label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "vit_mnist_improved_loss.png"))
plt.close()

# ---------------------------------------------------------------------------
# Avaliação final
# ---------------------------------------------------------------------------
_, accuracy = model.evaluate(x_test, y_test_oh, verbose=0)
print(f"\nTest accuracy: {round(accuracy * 100, 2)}%")
print("(baseline original tr02c.py: 99.10%)")
