import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello")
world = tf.constant("World")

print type(hello)
print hello

with tf.Session() as sess:
	result = sess.run(hello+world)
print(result)

a = tf.constant(10)
b = tf.constant(20)

with tf.Session() as sess:
	result = sess.run(a+b)
print result

const = tf.constant(10)

fill_mat = tf.fill((4, 4), 10)
my_zeros = tf.zeros((4, 4))
my_ones = tf.ones((4, 4))
myrandn = tf.random_normal((4, 4), mean = 0, stddev = 1.0)
myrandu = tf.random_uniform((4, 4), minval = 0, maxval = 1)

my_ops = [const, fill_mat, my_zeros, my_ones, myrandn, myrandu]

with tf.Session() as sess:
	for op in my_ops:
		print(sess.run(op))

a = tf.constant([[1, 2], 
				[3, 4]])
print a.get_shape()

b = tf.constant([[10], [100]])
print b.get_shape()

result = tf.matmul(a, b)
# sess = tf.InteractiveSession()
with tf.Session() as sess:
	print sess.run(result)
