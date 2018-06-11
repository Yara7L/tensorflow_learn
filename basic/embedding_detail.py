import tensorflow as tf 
import numpy as np 

# x=np.array([i for i in range(1,33)]).reshape([2,2,2,4])
# y=tf.nn.lrn(input=x,depth_radius=2,bias=0,alpha=1,beta=1)

# with tf.Session() as sess:
#     print(x)
#     print('=============')
#     print(y.eval())

sess = tf.InteractiveSession()
a=[[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]]
embedding = tf.Variable(a)
input_ids = tf.placeholder(dtype=tf.int32,shape=[None])
input_embedding = tf.nn.embedding_lookup(embedding,input_ids)

sess.run(tf.initialize_all_variables())
print(sess.run(embedding))
#[[1 0 0 0 0]
# [0 1 0 0 0]
# [0 0 1 0 0]
# [0 0 0 1 0]
# [0 0 0 0 1]]
print(sess.run(input_embedding,feed_dict={input_ids:[1,2,3,0,3,2,1]}))
# 26/(0+1*(25^2+26^2+27^2+28^2))^1