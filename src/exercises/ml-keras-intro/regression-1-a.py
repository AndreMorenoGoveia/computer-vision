#~/deep/algpi/densa/regression2/regression2b.py - 2025
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib
matplotlib.use("TkAgg")
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Define modelo de rede
model = Sequential()
model.add(Dense(2, activation='sigmoid', input_dim=2))
model.add(Dense(1, activation='linear'))

sgd=optimizers.SGD(learning_rate=0.2)
model.compile(optimizer=sgd, loss='mse')


AX = np.matrix('0.5 0.5; 0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0; 0.5 0.0; 0.0 0.5; 0.5 1.0; 1.0 0.5',dtype='float32')
AY = np.matrix('1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0', dtype='float32')



print("AX"); print(AX)
print("AY"); print(AY)
# As opcoes sao usar batch_size=2 ou 1
model.fit(AX, AY, epochs=1000, batch_size=2, verbose=2)
# Print the trained parameters
print("\nTrained MLP Parameters:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights: # Check if the layer has weights
        print(f"Layer: {layer.name}")
        print(f" Weights: \n{weights[0]}")
        print(f" Biases: \n{weights[1]}")


QX = np.matrix('1 0; 0 1; 0 0; 1 1',dtype='float32')
print("QX"); print(QX)
QP=model.predict(QX, verbose=2)
print("QP"); print(QP)
# Generate data for 3D plot
x1 = np.linspace(-0.5, 1.5, 50)
x2 = np.linspace(-0.5, 1.5, 50)
x1_grid, x2_grid = np.meshgrid(x1, x2)
QX_3d = np.hstack((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)))
QP_3d = model.predict(QX_3d)
# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, QP_3d.reshape(x1_grid.shape), cmap='viridis')
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('MLP Output')

plt.show()