"""
Extra Homework — Evaluate the cat-vs-dog classifier saved by cat_dog_treino.py.

Loads transf.keras, classifies the test set, reports the error rate, and
displays the first 10 misclassified cats and the first 10 misclassified dogs.
"""

import os
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')   # works even without a display; remove for interactive
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ---------------------------------------------------------------------------
# Locate test directories
# ---------------------------------------------------------------------------

BASE_DIR = 'cat_dog_clean'


def find_dir(parent, candidates):
    for name in candidates:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f'None of {candidates} found under {parent}')


test_dir = find_dir(BASE_DIR, ['test_set', 'test', 'Test'])
cat_dir = find_dir(test_dir, ['cats', 'cat'])
dog_dir = find_dir(test_dir, ['dogs', 'dog'])

cat_files = sorted(glob.glob(os.path.join(cat_dir, '*.jpg')))
dog_files = sorted(glob.glob(os.path.join(dog_dir, '*.jpg')))

print(f'Test cats: {len(cat_files)}   Test dogs: {len(dog_files)}')

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

if not os.path.exists('transf.keras'):
    raise FileNotFoundError('transf.keras not found — run cat_dog_treino.py first.')

model = keras.models.load_model('transf.keras')
print('Model loaded.')

NL, NC = 224, 224


def preprocess_file(path):
    img = keras_image.load_img(path, target_size=(NL, NC))
    arr = keras_image.img_to_array(img)
    return preprocess_input(np.expand_dims(arr, 0))[0]


# ---------------------------------------------------------------------------
# Predict in mini-batches to keep memory usage low
# ---------------------------------------------------------------------------

def predict_files(files, batch_size=32):
    preds = []
    for i in range(0, len(files), batch_size):
        batch_paths = files[i:i + batch_size]
        batch = np.stack([preprocess_file(p) for p in batch_paths])
        preds.extend(model.predict(batch, verbose=0).flatten().tolist())
        if (i // batch_size + 1) % 5 == 0:
            print(f'  {i + batch_size}/{len(files)} images processed')
    return np.array(preds)


print('\nEvaluating cats …')
cat_preds = predict_files(cat_files)

print('Evaluating dogs …')
dog_preds = predict_files(dog_files)

# ---------------------------------------------------------------------------
# Metrics
# The model outputs probability of being a dog (class 1, alphabetically second).
# cats  → true label 0,  predicted cat when pred ≤ 0.5
# dogs  → true label 1,  predicted dog when pred > 0.5
# ---------------------------------------------------------------------------

cat_correct = np.sum(cat_preds <= 0.5)
dog_correct = np.sum(dog_preds > 0.5)
total = len(cat_files) + len(dog_files)
correct = cat_correct + dog_correct
accuracy = correct / total
error_rate = 1.0 - accuracy

print(f'\n{"─" * 40}')
print(f'Total test images : {total}')
print(f'Correct           : {correct}')
print(f'Test accuracy     : {accuracy * 100:.2f} %')
print(f'Error rate        : {error_rate * 100:.2f} %')
print(f'{"─" * 40}')

# Misclassified files (sorted to get "first 10" in alphabetical order)
wrong_cats = [f for f, p in zip(cat_files, cat_preds) if p > 0.5]   # cat → dog
wrong_dogs = [f for f, p in zip(dog_files, dog_preds) if p <= 0.5]  # dog → cat

print(f'Misclassified cats (predicted dog): {len(wrong_cats)}')
print(f'Misclassified dogs (predicted cat): {len(wrong_dogs)}')

# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def plot_errors(files, title, max_show=10, save_name=None):
    show = files[:max_show]
    if not show:
        print(f'No errors in: {title}')
        return

    cols = min(5, len(show))
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 0.5))
    axes = np.array(axes).flatten()

    for ax, path in zip(axes, show):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(os.path.basename(path), fontsize=7)
        ax.axis('off')
    for ax in axes[len(show):]:
        ax.axis('off')

    plt.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()

    out = save_name or title.replace(' ', '_').lower() + '.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


plot_errors(wrong_cats,
            'Misclassified cats (predicted as dog)',
            save_name='wrong_cats.png')
plot_errors(wrong_dogs,
            'Misclassified dogs (predicted as cat)',
            save_name='wrong_dogs.png')
