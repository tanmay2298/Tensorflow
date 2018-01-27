import tensorflow as tf

sess = tf.InteractiveSession()

my_tensor = tf.random_uniform((4, 4), 0, 1)
print my_tensor

my_var = tf.Variable(initial_value = my_tensor)

print my_var

init = tf.global_variables_initializer()
sess.run(init)
print sess.run(my_var)

ph = tf.placeholder(tf.float32, shape = (None, 5))
print ph
