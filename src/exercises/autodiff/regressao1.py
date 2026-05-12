import tensorflow as tf

x1 = tf.constant(0.6)
x2 = tf.constant(-0.3)
y  = tf.constant(1.0)

w1 = tf.Variable(-0.2)
w2 = tf.Variable(0.5)
w3 = tf.Variable(0.9)
w4 = tf.Variable(-0.6)
w5 = tf.Variable(0.2)
w6 = tf.Variable(-0.4)
b1 = tf.Variable(0.4)
b2 = tf.Variable(-0.2)
b3 = tf.Variable(-0.5)

with tf.GradientTape() as tape:
    z1 = w1*x1 + w2*x2 + b1
    p1 = tf.math.sigmoid(z1)
    z2 = w3*x1 + w4*x2 + b2
    p2 = tf.math.sigmoid(z2)
    z3 = w5*p1 + w6*p2 + b3
    p3 = tf.math.sigmoid(z3)
    c  = (p3 - y)**2

grads = tape.gradient(c, [w1, w2, w3, w4, w5, w6, b1, b2, b3])

print("dcdw1:", grads[0].numpy())
print("dcdw2:", grads[1].numpy())
print("dcdw3:", grads[2].numpy())
print("dcdw4:", grads[3].numpy())
print("dcdw5:", grads[4].numpy())
print("dcdw6:", grads[5].numpy())
print("dcdb1:", grads[6].numpy())
print("dcdb2:", grads[7].numpy())
print("dcdb3:", grads[8].numpy())
