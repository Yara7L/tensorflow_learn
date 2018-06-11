import tensorflow as tf
import numpy as numpy
'''
# a easy linear model
x_data=numpy.float32(numpy.random.rand(2,100))
y_data=numpy.dot([0.100,0.200],x_data)+0.300

b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(W,x_data)+b

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for step in range(0,201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(W),sess.run(b))

# hello,TensorFlow
hello=tf.constant('Hello,TensorFlow!')
sess=tf.Session()
print(sess.run(hello))

#the first example
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
product=tf.matmul(matrix1,matrix2)

sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result=sess.run(product)
    print(result)

#TnteractiveSession
sess=tf.InteractiveSession()
x=tf.Variable([1.0,2.0])
a=tf.constant([3.0,3.0])
x.initializer.run()
sub=tf.subtract(x,a)
print(sub.eval())

#Variables
state=tf.Variable(0,name="counter")

one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init_op=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

#Fetch
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
intermed=tf.add(input2,input3)
mul=tf.multiply(input1,intermed)
with tf.Session() as sess:
    result=sess.run([mul,intermed])
    print(result)

#Feed,临时代替途中的任意操作中的tensor可以对任何操作提交不定，直接插入tensor
imput1=tf.placeholder(tf.float32)
imput2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:7.,input2:2.}))
'''

train_X=numpy.linspace(-1,1,100)
train_Y=2*train_X+numpy.random.randn(*train_X.shape)*0.33+10

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
w=tf.Variable(0.0,name="weight")
b=tf.Variable(0.0,name="bias")
loss=tf.square(Y-tf.multiply(X,w)-b)
train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch=1
    for i in range(10):
        for (x,y) in zip(train_X,train_Y):
            _,w_value,b_value=sess.run([train_op,w,b],feed_dict={X:x,Y:y})
        print("Epoch:{},w:{},b:{}".format(epoch,w_value,b_value))
        epoch+=1