"""http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html"""

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

"""
x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，
每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）
"""
x = tf.placeholder("float", [None, 784])

"""
我们赋予tf.Variable不同的初值来创建不同的Variable：
在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
这就是要做的模型
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)
"""
交叉熵是用来衡量我们的预测用于描述真相的低效性。
为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：y_
"""
y_ = tf.placeholder("float", [None, 10])


"""
计算交叉熵:
"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
"""
tf.argmax(y, 1)：返回的是模型对于任一输入x预测到的标签值；
tf.argmax(y_, 1)：代表正确的标签
tf.equal() 检测我们的预测是否真实标签匹配
"""

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
"""
把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
