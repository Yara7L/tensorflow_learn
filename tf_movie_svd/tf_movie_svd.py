import time
from collections import deque
import numpy as np 
import tensorflow as tf 
from six import next
from tensorflow.core.framework import summary_pb2
import data
import ops

np.random.seed(13575)

batch_size=1000
user_num=6040
item_num=3952
dim=15
epoch_max=100
device="/cpu:0"

def clip(x):
    return np.clip(x,1.0,5.0)

def make_scalar_summary(name,val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,simple_value=val)])

def get_data():
    df=data.read_process("E:/dataset/NLP/movies/ml-1m/ratings.dat",sep="::")
    rows=len(df)
    df=df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index=int(rows*0.9)
    df_train=df[0:split_index]
    df_test=df[split_index:].reset_index(drop=True)
    return df_train,df_test

def svd(train,test):
    samples_per_batch=len(train)//batch_size
    iter_train=data.ShuffleIterator([train["user"],train["item"],train["rate"]],batch_size=batch_size)

    iter_test=data.OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=-1)

    user_batch=tf.placeholder(tf.int32,shape=[None],name="id_user")
    item_batch=tf.placeholder(tf.int32,shape=[None],name="id_item")
    rate_batch=tf.placeholder(tf.int32,shape=[None])

    infer,regularizer=ops.inference_svd(user_batch,item_batch,user_num=user_num,item_num=item_num,dim=dim,device=device)
    _,train_op=ops.optimization(infer,regularizer,rate_batch,learning_rate=0.001,reg=0.05,device=device)

    init_op=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer=tf.summary.FileWriter(logdir="/tmp/svd/log",graph=sess.graph)
        print("{} {} {} {}".format("epoch","train_error","val_error","elapsed_time"))
        errors=deque(maxlen=samples_per_batch)
        start=time.time()
        for i in range(epoch_max*samples_per_batch):
            users,items,rates=next(iter_train)
            _,pre_batch=sess.run([train_op,infer],feed_dict={user_batch:users,item_batch:items,rate_batch:rates})

            pred_barch=clip(pred_barch)
            errors.append(np.power(pre_batch-rates,200))
            if i %samples_per_batch==0:
                train_err=np.sqrt(np.mean(errors))
                test_err2=np.array([])
                for users,items,rates in iter_test:
                    pred_barch=sess.run(infer,feed_dict={user_batch:user,item_batch:items})
                    pred_barch=clip(pred_barch)
                    test_err2=np.append(test_err2,np.power(pred_barch-rates,2))
                end=time.time()
                test_err=np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end


if __name__ == '__main__':
    df_train, df_test = get_data()
    svd(df_train, df_test)
    print("Done!")

