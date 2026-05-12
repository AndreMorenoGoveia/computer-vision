import tensorflow as tf

x11 = tf.constant(0.8)
x12 = tf.constant(-0.1)
y1  = tf.constant(1.0)

x21 = tf.constant(-0.1)
x22 = tf.constant(0.9)
y2  = tf.constant(0.0)

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
    z11 = w1*x11 + w2*x12 + b1
    p11 = tf.math.sigmoid(z11)
    z21 = w3*x11 + w4*x12 + b2
    p21 = tf.math.sigmoid(z21)
    z31 = w5*p11 + w6*p21 + b3
    p31 = tf.math.sigmoid(z31)
    c1  = (p31 - y1)**2

    z12 = w1*x21 + w2*x22 + b1
    p12 = tf.math.sigmoid(z12)
    z22 = w3*x21 + w4*x22 + b2
    p22 = tf.math.sigmoid(z22)
    z32 = w5*p12 + w6*p22 + b3
    p32 = tf.math.sigmoid(z32)
    c2  = (p32 - y2)**2

    c = (c1 + c2) / 2

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
