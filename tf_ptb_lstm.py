import tensorflow as tf 
import time 
import numpy as np
# 没有reader包
# from tensorflow.models.rnn.ptb import reader

flags=tf.flags
logging=tf.logging

flags.DEFINE_string(
    "model","small","A type of model. Possible options are:small,medium,large.")

flags.DEFINE_string(
    "data_path",'E:/dataset/RNN/LSTM/simple-examples/data/',"data_path")

flags.DEFINE_bool("use_fp16",False,"Train using 16-bit floats instead of 32-bit floats")

FLAGS=flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBModel(object):
    def __init__(self,is_training,config):
        '''
        is_training=False,不会进行参数的修正
        '''
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        size=config.hidden_size
        vocab_size=config.vocab_size

        # 输入输出，两个都是index序列
        self._input_data=tf.placeholder(tf.init32,[batch_size,num_steps])
        self._targets=tf.placeholder(tf.int32,[batch_size,num_steps])

        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)
        # c,h为输出,state_is_tuple=True,state=(c,h);为False，拼接的张量，state=tf.concat(1,[c,h]).运行时，返回state和h
        if is_training and config.config.keep_prob <1:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,output_keep_prob=config.keep_prob)
        # cnn与rnn不同，rnn在t-1时刻到t时刻进行计算，中间不进行memory的dropout；仅在同一时刻中，多层cell之间进行dropout
        # input_keep_prob,output_keep_prob控制输入输出的dropout概率
        # cell是一个多层结构了。把每一层的lstm cell连在一起得到多层的RNN？？？

        cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers,state_is_tuple=True)

        self._initial_state=cell.zero_state(batch_size,data_type())

        with tf.device("/cpu:0"):
            embedding=tf.get_variable(
                "embedding",[vocab_size,size],dtype=data_type())
            inputs=tf.nn.embedding_lookup(embedding,self._input_data)
        
        if is_training and config.keep_prob<1:
            inputs=tf.nn.dropout(inputs,config.keep_prob)
        # keep_prob<1,需要对输入进行dropout，跟rnn的dropout有所不同。

        outputs=[]
        state=self._initial_state#各个batch中的状态
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                    #必不可少，不然会报错，同一命名域内不允许存在多个同一名字变量的原因,RNN时间序列
                # 从state开始运行RNN架构，输出为cell的输出以及新的state
                (cell_output,state)=cell(inputs[:,time_step,:],state) #按顺序向cell输入文本数据
                outputs.append(cell_output)#output:shape[num_steps][batch,hidden_size]
      
        # 将list展开，成[batch,hidden_size*num_steps],然后reshape,成[batch*num_steps,hidden_size]
        output=tf.reshape(tf.concat(1,outputs),[-1,size])

        # 多层lstm输出转化成one-hot表示的向量
        softmax_w=tf.get_variable("softmax_w",[size,vocab_size],dtype=data_type())
        softmax_b=tf.get_variable("softmax_b",[vocab_size],dtype=data_type())        
        # [batch*num_steps,vocab_size]从隐藏语义转化成完全表示
        logits=tf.matmul(output,softmax_w)+softmax_b

        # 损失函数
        loss=tf.nn.seq2seq.sequence_loss_by_example(
            [logits],#output [batch*num_steps,vocab_size]
            [tf.reshape(self._targets,[-1])],#target,[batch_size,num_steps],展开成一维list
            [tf.ones([batch_size*num_steps],dtype=data_type())])#weight
        self._cost=cost=tf.reduce_sum(loss)/batch_size #计算得到平均每批batch的误差
        self._final_state=state

        if not is_training:
            return
        
        self._lr=tf.Variable(0.0,trainable=False)
        tvars=tf.trainable_variables()
        # 梯度衰减，根据张量间的和的norm来clip多个张量

        # clip_by_global_norm: 梯度衰减，具体算法为t_list[i] * clip_norm / max(global_norm, clip_norm)
        # 这里gradients求导，ys和xs都是张量
        # 返回一个长为len(xs)的张量，其中的每个元素都是\grad{\frac{dy}{dx}}
        # clip_by_global_norm 用于控制梯度膨胀,前两个参数t_list, global_norm, 则
        # t_list[i] * clip_norm / max(global_norm, clip_norm)
        # 其中 global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),config.max_grad_norm)

        optimizer=tf.train.GradientDescentOptimizer(self._lr)
        # 将梯度用于变量
        self._train_op=optimizer.apply_gradients(zip(grads,tvars))

        # 外部向graph输入新的lr,new_lr来更新lr
        self._new_lr=tf.placeholder(
            tf.float32,shape=[],name="new_learning_rate"
        )
        self._lr_update=tf.assign(self._lr,self._new_lr)

    def assign_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value})
    
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op    


class SmallConfig(object):

    init_scale = 0.1        # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
    learning_rate = 1.0     # 学习速率,在文本循环次数超过max_epoch以后会逐渐降低
    max_grad_norm = 5       # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
    num_layers = 2          # lstm层数
    num_steps = 20          # 单个数据中，序列的长度。
    hidden_size = 200       # 隐藏层中单元数目
    max_epoch = 4           # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
    max_max_epoch = 13      # 指的是整个文本循环次数。
    keep_prob = 1.0         # 用于dropout.每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
    lr_decay = 0.5          # 学习速率衰减
    batch_size = 20         # 每批数据的规模，每批有20个。
    vocab_size = 10000      # 词典规模，总共10K个词

class MediumConfig(object):
    init_scale=0.05
    learning_rate=1.0
    max_grad_norm=5
    num_layers=2
    num_steps=35
    hidden_size=650
    max_epoch=6
    max_max_epoch=39
    keep_prob=0.5
    lr_decay=0.8
    batch_size=20  
    vocab_size=10000

class LargeConfig(object):
    init_scale=0.04
    learning_rate=1.0
    max_grad_norm=10
    num_layers=2
    num_steps=35
    hidden_size=1500
    max_epoch=14
    max_max_epoch=55
    keep_prob=0.35
    lr_decay=1/1.15
    batch_size=20  
    vocab_size=10000

class TestConfig(object):
    init_scale=0.1
    learning_rate=1.0
    max_grad_norm=1
    num_layers=1
    num_steps=2
    hidden_size=2
    max_epoch=1
    max_max_epoch=1
    keep_prob=1.0
    lr_decay=0.5
    batch_size=20  
    vocab_size=10000

def run_epoch(session,model,data,eval_op,verbose=False):
    epoch_size=((len(data)//model.batch_size)-1)//model.num_steps
    start_time=time.time()
    costs=0.0
    iters=0
    state=session.run(model.initial_state)
    for step,(x,y) in enumerate(reader.ptb_iterator(data,model.batch_size,model.num_steps)):
        fetches=[model.cost,model.final_state,eval_op]
        feed_dict={}
        feed_dict[model.input_data]=x
        feed_dict[model.targets]=y
        for i,(c,h) in enumerate(model.initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        # cost,state,_=session.run([m.cost,m.final_state,eval_op],
        #                         {m.input_data:x,
        #                         m.targets:y,
        #                         m.initial_state:state})
        cost,state,_=session.run(fetches,feed_dict)
        costs+=cost
        iters+=model.num_steps

        if verbose and step %(epoch_size//10)==10:  
            print("%.3f perplexity:%.3f speed:%.0f wps"%
                (step*1.0/epoch_size,np.exp(costs/iters),
                iters*model.batch_size/(time.time()-start_time)))

    return np.exp(costs/iters)

def get_config():
    if FLAGS.model=="small":
        return SmallConfig()
    elif FLAGS.model=="medium":
        return SmallConfig()
    elif FLAGS.model=="large":
        return SmallConfig()
    elif FLAGS.model=="test":
        return TestConfig()
    else:
        raise ValueError("Invalid model:%s",FLAGS.model)

if __name__=="__main__":
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    print(FLAGS.data_path)

    raw_data=reader.ptb_raw_data(FLAGS.data_path)
    train_data,valid_data,test_data,_=raw_data

    config=get_config()
    eval_config=get_config()
    eval_config.batch_size=1
    eval_config.num_steps=1

    with tf.Graph().as_default(),tf.Session() as session:
        initializer=tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            m=PTBModel(is_training=True,config=config)#train
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            mvalid=PTBModel(is_training=False,config=config)#valid
            mtest=PTBModel(is_training=False,config=eval_config)#test

        summary_writer=tf.train.SummaryWriter('E:/ML/.vscode/logs/peen_treebank/lstm_logs',session.graph)

        tf.global_variables_initializer().run()

        for i in range(config.max_max_epoch):#所有文本重复多次进入模型训练
            lr_decay=config.lr_decay**max(i-config.max_epoch,0.0)
            m.assign_lr(session,config.learning_rate*lr_decay)

            print("Epoch:%d Learning rate:%.3f"%(i+1,session.run(m.lr)))
            train_perplexity=run_epoch(session,m,train_data,m.train_op,verbose=True)
            print("Epoch:%d Train Perplexity:%.3f"%(i+1,train_perplexity))
            valid_perplexity=run_epoch(session,mvalid,valid_data,tf.no_op())
            print("Epoch:%d Valid Perplexity:%.3f"%(i+1,valid_perplexity))

        test_perplexity=run_epoch(session,mtest,test_data,tf.no_op())
        print("Test Perplexity:%.3f"%test_perplexity) 