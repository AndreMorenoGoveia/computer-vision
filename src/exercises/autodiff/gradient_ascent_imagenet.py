import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

# ImageNet class indices (0-based) for VGG16/Keras
TARGET_CLASSES = {
    "chimpanzee": 367,
    "tiger":       292,
    "flamingo":    130,
    "volcano":     980,
    "coral_reef":  973,
    "strawberry":  949,
}

IMG_SIZE    = 224
ITERATIONS  = 300
LR          = 2.0
OUTPUT_DIR  = "gradient_ascent_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading VGG16...")
model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=True,
    classifier_activation=None,
)
model.trainable = False
print("Model loaded.\n")


def gradient_ascent(class_idx, class_name):
    print(f"Generating: {class_name} (class index {class_idx})")

    # Random starting image in [0.4, 0.6] — neutral grey range
    img = tf.Variable(
        np.random.uniform(0.4, 0.6, (1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    )

    for step in range(ITERATIONS):
        with tf.GradientTape() as tape:
            x = tf.keras.applications.vgg16.preprocess_input(img * 255.0)
            logits = model(x, training=False)
            loss = logits[0, class_idx]

        grads = tape.gradient(loss, img)
        # Normalise so step size is consistent regardless of gradient magnitude
        grads /= tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-8
        img.assign_add(grads * LR)
        img.assign(tf.clip_by_value(img, 0.0, 1.0))

        if (step + 1) % 100 == 0:
            prob = tf.nn.softmax(logits)[0, class_idx].numpy()
            print(f"  step {step+1:3d}/{ITERATIONS}  probability = {prob:.4f}")

    return img.numpy()[0]


def top5(img_array):
    x = tf.keras.applications.vgg16.preprocess_input(
        img_array[np.newaxis] * 255.0
    )
    logits = model.predict(x, verbose=0)
    return tf.keras.applications.vgg16.decode_predictions(logits, top=5)[0]


# ── Run gradient ascent for each class ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, idx) in zip(axes, TARGET_CLASSES.items()):
    img = gradient_ascent(idx, name)

    print(f"  Top-5 predictions after gradient ascent:")
    for _, label, prob in top5(img):
        print(f"    {label:25s}  {prob:.4f}")

    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    Image.fromarray((img * 255).astype(np.uint8)).save(path)

    ax.imshow(img)
    ax.set_title(name)
    ax.axis("off")
    print()

plt.suptitle("Gradient ascent — images that maximise ImageNet class scores", fontsize=11)
plt.tight_layout()
grid_path = os.path.join(OUTPUT_DIR, "all_classes.png")
plt.savefig(grid_path, dpi=150)
print(f"Grid saved to {grid_path}")
print("Done!")
