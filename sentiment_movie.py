import numpy as np 
import tensorflow as tf 
import random
import pickle
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

pos_file='E:/dataset/NLP/Sentiment/rt-polaritydata/pos.txt'
neg_file='E:/dataset/NLP/Sentiment/rt-polaritydata/neg.txt'


def create_dic(pos_file,neg_file):
    dic=[]
    def process_file(file):
        with open(file,'r',encoding='utf-8') as f:
            dic=[]
            lines=f.readlines()
            for line in lines:
                words=word_tokenize(line.lower())  #令牌化
                dic+=words
            return dic
    dic+=process_file(pos_file)
    dic+=process_file(neg_file)

    lemmatizer=WordNetLemmatizer()
    dic=[lemmatizer.lemmatize(word) for word in dic] #将单词还原成原级形式
    
    word_count=Counter(dic) #计算单词出现次数
    
    dic=[]
    for word in word_count:
        if word_count[word]<2000 and word_count[word]>20: #去掉常用词和及不常用词
            dic.append(word)
    return dic   

dic=create_dic(pos_file,neg_file)
print(len(dic))


def normalize_dataset(dic):
    dataset=[]
    def string_to_vector(dic,review,clf):
        words=word_tokenize(line.lower())
        lemmatizer=WordNetLemmatizer()
        words=[lemmatizer.lemmatize(word) for word in words]

        features=np.zeros(len(dic))
        for word in words:
            if word in dic:
                features[dic.index(word)]=1
        return np.array([features,clf])

    with open(pos_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            one_sample=string_to_vector(dic,line,[1,0])
            dataset.append(one_sample)
           
    with open(neg_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            one_sample=string_to_vector(dic,line,[0,1])
            dataset.append(one_sample)
    
    return dataset

dataset=normalize_dataset(dic)

random.shuffle(dataset)

'''
# 整理好数据保存到文件。
with open('E:/dataset/NLP/Sentiment/rt-polaritydata/save.pickle','wb') as f:
    pickle.dump(dataset,f)
'''

test_size=int(len(dataset)*0.2)

dataset=np.array(dataset)

train_dataset=dataset[:-test_size]
test_dataset=dataset[-test_size:]

n_input_layer=len(dic)

n_layer_1=1064
n_layer_2=2000

n_output_layer=2

def neural_network(data):
    layer_1_w_b={'w_':tf.Variable(tf.random_normal([n_input_layer,n_layer_1])),'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b={'w_':tf.Variable(tf.random_normal([n_layer_1,n_layer_2])),'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b={'w_':tf.Variable(tf.random_normal([n_layer_2,n_output_layer])),'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1=tf.add(tf.matmul(data,layer_1_w_b['w_']),layer_1_w_b['b_'])
    layer_1=tf.nn.relu(layer_1)
    layer_2=tf.add(tf.matmul(data,layer_2_w_b['w_']),layer_2_w_b['b_'])
    layer_2=tf.nn.relu(layer_2)
    layer_output=tf.add(tf.matmul(layer_2,layer_output_w_b['w_']),layer_output_w_b['b_'])

    return layer_output

batch_size=50

X=tf.placeholder('float',[None,len(train_dataset[0][0])])
Y=tf.placeholder('float')

def train_neural_network(X,Y):
    predict=neural_network(X)
    cost_func=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=Y))
    optimizer=tf.train.AdamOptimizer().minimize(cost_func)

    epochs=20
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss=0

        i=0
        random.shuffle(train_dataset)

        train_x=dataset[:,0]
        train_y=dataset[:,1]

        for epoch in range(epochs):
            while i<len(train_x):
                start=1
                end=i+batch_size

                batch_x=train_x[start:end]
                batch_y=train_y[start:end]

                _,c=session.run([optimizer,cost_func],feed_dict={X:list(batch_x),Y:list(batch_y)})
                epoch_loss+=c
                i+=batch_size
            print(epoch,':',epoch_loss)
        
        test_x=test_dataset[:,0]
        test_y=test_dataset[:,1]
        correct=tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('准确率：',accuracy.eval({X:list(test_x),Y:list(test_y)}))

train_neural_network(X,Y)

   