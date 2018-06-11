import tensorflow as tf 
import numpy as np
import collections
import pickle as pkl 
from pprint import pprint
from pymongo import MongoClient
import re
import jieba
import os.path as path
import os
import math

class word2vec():
    def __init__(self,vocab_list=None,embedding_size=200,win_len=3,
                num_sampled=1000,learning_rate=1.0,logdir='/tmp/simp_word2vec',model_path=None,):
        self.batch_size=None
        if model_path!=None:
            self.load_model(model_path)
        else:
            assert type(vocab_list)==list
            self.vocab_list =vocab_list
            self.vocab_size =vocab_list.__len__()
            self.embedding_size=embedding_size
            self.win_len=win_len
            self.num_sampled=num_sampled
            self.learning_rate=learning_rate
            self.logdir=logdir

            # word=>id的映射
            self.word2id={}
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]]=i
            
            # train parameters
            self.train_words_num=0  #单词数
            self.train_sents_num=0  #句子数
            self.train_times_num=0  #训练次数

            #train loss records
            self.train_loss_records=collections.deque(maxlen=10)
            self.train_loss_k10=0
        
        self.build_graph()
        self.init_op()
        if model_path!=None:
            tf_model_path=os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_model_path)
    
    def init_op(self):
        self.sess=tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer=tf.summary.FileWriter(self.logdir,self.graph)

    def build_graph(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.train_inputs=tf.placeholder(tf.int32,shape=[self.batch_size])
            self.train_labels=tf.placeholder(tf.int32,shape=[self.batch_size,1])

            # 词向量矩阵，初始时为均匀随机正太分布
            self.embedding_dict=tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
            )

            self.nce_weight=tf.Variable(tf.truncated_normal([self.vocab_size,self.embedding_size],stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases=tf.Variable(tf.zeros([self.vocab_size]))

            # 输入序列向量化
            embed=tf.nn.embedding_lookup(self.embedding_dict,self.train_inputs)

            # noise-contrastice-estimation
            self.loss=tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=self.num_sampled, #负采样的个数
                    num_classes=self.vocab_size  #类别个数
                )
            )

            
            tf.summary.scalar('loss',self.loss)

            self.train_op=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)

            #计算指定若干单词的相似度
            self.test_word_id=tf.placeholder(tf.int32,shape=[None])
            vec_l2_model=tf.sqrt(
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True)
            )

            avg_l2_model=tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_l2_model',avg_l2_model)

            self.normed_embedding=self.embedding_dict/vec_l2_model
            test_embed=tf.nn.embedding_lookup(self.normed_embedding,self.test_word_id)
            self.similarity=tf.matmul(test_embed,self.normed_embedding,transpose_b=True)

            self.init=tf.global_variables_initializer()
            self.mergerd_summary_op=tf.summary.merge_all()
            self.saver=tf.train.Saver()

    def train_by_sentence(self,input_sentence=[]):
        sent_num=input_sentence.__len__()
        batch_inputs=[]
        batch_labels=[]
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start=max(0,i-self.win_len)  #窗口为[-win_len,+win_len],总长为2*win_len+1
                end=min(sent.__len__(),i+self.win_len+1)
                #某个单词对应窗口中的其他单词转化为id计入label，该单词本身计入input
                for index in range(start,end):
                    if index==i:
                        continue 
                    else:
                        input_id=self.word2id.get(sent[i])
                        label_id=self.word2id.get(sent[index])
                        if not (input_id and label_id):#如果单词不在词典中，则跳过
                            continue 
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs)==0:
            return
        
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])

        feed_dict={
            self.train_inputs:batch_inputs,
            self.train_labels:batch_labels
        }
        
        _,loss_val,summary_str=self.sess.run([self.train_op,self.loss,self.mergerd_summary_op],feed_dict=feed_dict)

        # trian loss
        self.train_loss_records.append(loss_val)
        self.train_loss_k10=np.mean(self.train_loss_records)
        if self.train_sents_num % 1000==0:
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("{a} sentences dealed, loss:{b}".format(a=self.train_sents_num,b=self.train_loss_k10))
        
        #train times
        self.train_words_num+=batch_inputs.__len__()
        self.train_sents_num+=input_sentence.__len__()
        self.train_times_num+=1

    def cal_similarity(self,test_word_id_list,top_k=10):
        sim_matrix=self.sess.run(self.similarity,feed_dict={self.test_word_id:test_word_id_list})
        sim_mean=np.mean(sim_matrix)
        sim_var=np.mean(np.square(sim_matrix-sim_mean))
        test_words=[]
        near_words=[]
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id=(-sim_matrix[i,:].argsort()[1:top_k+1])
            nearst_word=[self.vocab_list[x] for x in nearst_id]
            near_words.append(near_words)
        return test_words,near_words,sim_matrix,sim_var
    
    def save_model(self,save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        model={}
        var_names=['vocab_size','vocab_list','learning_rate','word2id','embedding_size','logdir',
                    'win_len','num_sampled','train_words_num','train_sents_num','train_times_num','train_loss_records','train_loss_k10']

        for var in var_names:
            model[var]=eval('self.'+var)
        
        param_path=os.path.join(save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as f:
            pkl.dump(model,f)

        tf_path = os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)

    def load_model(self,model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path = os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as f:
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']

if __name__=='__main__':
    #读取停用词
    stop_words=[]
    with open('E:/dataset/NLP/word2vec/stop_words.txt','r',encoding='utf-8') as f:
        line=f.readline()
        while line:
            stop_words.append(line[:-1])
            line=f.readline()
    stop_words=set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    #读取文本，预处理，分词，得到词典
    raw_word_list=[]
    sentence_list=[]
    with open('E:/dataset/NLP/word2vec/280.txt','r',encoding='gbk') as f:
        line=f.readline()
        while line:
            while '\n' in line:
                line=line.replace('\n','')
            while ' ' in line:
                line=line.replace(' ','')
            if len(line)>0:
                raw_words=list(jieba.cut(line,cut_all=False))
                dealed_words=[]
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520','www','com','http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line=f.readline()
    word_count=collections.Counter(raw_word_list)
    print('文本中总共由{n1}个单词，不重复单词数{n2},选取前30000个单词进入词典'
        .format(n1=len(raw_word_list),n2=len(word_count)))
    word_count=word_count.most_common(30000)
    word_list=[x[0] for x in word_count]

    w2v=word2vec(vocab_list=word_list,embedding_size=200,win_len=2,learning_rate=1,num_sampled=100,logdir='tmp/280')
    
    num_steps=100000
    for i in range(num_steps):
        sent=sentence_list[i%len(sentence_list)]
        w2v.train_by_sentence([sent])

    w2v.save_model('E:/ML/models/nlp/word2vec')

    test_word=['萧炎','灵魂','火焰','长老','尊者','皱眉']
    test_id=[word_list.index(x) for x in test_word]

    print(w2v.cal_similarity(test_id,top_k=1))