'''
tensorflow的基本概念应用
mnist的线性拟合
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_dir','/tmp/data/','directory for storing data')
mnist=input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
a=tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(a),reduction_indices=[1]))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(cross_entropy)

# equal()bool类型0，1；cast()转化为float32，计算均值得分。
correct_prediction=tf.equal(tf.argmax(a,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Session V.S. InteractiveSession
# 前者启动session前构建整个计算图，再启动；后者在可以在运行图时，加入一些计算图
# with...as...后可以不使用close，调用InteractiveSession后面需close
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train.run({x:batch_xs,y:batch_ys})
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

