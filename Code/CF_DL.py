#*******************************Improvment_Method********************************#
#In this file, I will try new method: CF_as_Deep_Learning(AutoRec).
#AutoRec: autoencoders meets recommendation system
#NOTICE: In this method, the class autorec is refered from https://github.com/XiuzeZhou/Recommender-Systems/blob/master/autorec.py
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

ratings_loc_1 = 'data/ratings_mini_1.csv'#750
ratings_1 = pd.read_csv(ratings_loc_1, index_col=None)
ratings_loc_2 = 'data/ratings_mini_2.csv'#2500
ratings_2 = pd.read_csv(ratings_loc_2, index_col=None)
ratings_loc_3 = 'data/ratings_mini.csv'#125
ratings_3 = pd.read_csv(ratings_loc_3, index_col=None)
ratings_loc_4 = 'data/ratings.csv'
ratings_4 = pd.read_csv(ratings_loc_4, index_col=None)

class autorec():
    def __init__(self, users_num = None, items_num = None, hidden_size = 250, \
        batch_size = 256, learning_rate = 1e-3, lamda_regularizer = 1e-3):
        self.users_num = users_num
        self.items_num = items_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        
        self.train_loss_records = []  
        self.build_graph()

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():      
            self.rating_inputs = tf.compat.v1.placeholder(tf.float32, shape = [None, self.items_num], name='rating_inputs')
            self.weights = self._initialize_weights()        
            self.y_ = self.inference(rating_inputs=self.rating_inputs)
            self.loss_train = self.loss_function(true_r=self.rating_inputs, predicted_r=self.y_, lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 
            self.predictions = self.inference(rating_inputs=self.rating_inputs)
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    def _init_session(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['V'] = tf.Variable(tf.random.normal([self.items_num, self.hidden_size], 0.0, 0.1), name='V')
        all_weights['mu'] = tf.Variable(tf.zeros([self.hidden_size]), name='mu')
        all_weights['W'] = tf.Variable(tf.random.normal([self.hidden_size, self.items_num], 0.0, 0.1), name='W')
        all_weights['b'] = tf.Variable(tf.zeros([self.items_num]), name='b')
        return all_weights
    
    def train(self, data_mat):
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size) % instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.rating_inputs: data_mat[start:end]}  
            loss, opt = self.sess.run([self.loss_train, self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records

    def inference(self, rating_inputs):
        encoder = tf.nn.sigmoid(tf.matmul(rating_inputs, self.weights['V']) + self.weights['mu'])
        decoder = tf.identity(tf.matmul(encoder, self.weights['W']) + self.weights['b'])
        return decoder         
        
    def loss_function(self, true_r, predicted_r, lamda_regularizer=1e-3):
        idx = tf.where(true_r>0)
        true_y = tf.gather_nd(true_r, idx)
        predicted_y = tf.gather_nd(predicted_r, idx)
        mse = tf.compat.v1.losses.mean_squared_error(true_y, predicted_y)
        regularizer = tf.keras.regularizers.l2(lamda_regularizer)
        regularization = regularizer(self.weights['V']) + regularizer(self.weights['W'])
        cost = mse + regularization
        return cost 
    
    def predict_ratings(self, data_mat):
        pred_mat = np.zeros([self.users_num, self.items_num])
        
        instances_size = len(data_mat)
        batch_size = self.batch_size
        total_batch = math.ceil(instances_size/batch_size)
        for batch in range(total_batch):
            start = (batch*batch_size)% instances_size
            end = min(start+batch_size, instances_size)
            feed_dict = {self.rating_inputs: data_mat[start:end]}  
            out = self.sess.run([self.predictions], feed_dict=feed_dict)
            pred_mat[start:end,:] = np.reshape(out,(-1,self.items_num))

        return pred_mat

def AutoRec(data, base_type = 'user_based', hidden_size = 250, batch_size = 256, lamda_regularizer = 1e-4, \
    learning_rate = 1e-3, epoches = 100):
    ratings = data
    fix_n = len(ratings['userId'].unique())
    fix_m = len(ratings['movieId'].unique())
    user_dict = {}
    movie_dict = {}
    train_list, test_list = train_test_split(ratings,test_size=0.2)
    average = train_list['rating'].mean()

    def init_R(source,user_dict,movie_dict):
        countnum_user = -1
        countnum_movie = -1
        n = max(len(source['userId'].unique()),fix_n)
        m = max(len(source['movieId'].unique()),fix_m)
        R = np.zeros((n,m),dtype=float)
        for index,item in source.iterrows():
            if int(item['userId']) not in user_dict.keys():
                countnum_user += 1
                user_dict[int(item['userId'])] = countnum_user
                if int(item['movieId']) not in movie_dict.keys():
                    countnum_movie += 1
                    movie_dict[int(item['movieId'])] = countnum_movie
                    R[user_dict[int(item['userId'])]][movie_dict[int(item['movieId'])]] = item['rating']
                else:
                    R[user_dict[int(item['userId'])]][movie_dict[int(item['movieId'])]] = item['rating']
            else:
                if int(item['movieId']) not in movie_dict.keys():
                    countnum_movie += 1
                    movie_dict[int(item['movieId'])] = countnum_movie
                    R[user_dict[int(item['userId'])]][movie_dict[int(item['movieId'])]] = item['rating']
                else:
                    R[user_dict[int(item['userId'])]][movie_dict[int(item['movieId'])]] = item['rating']
        return R

    train_R = init_R(train_list,user_dict,movie_dict)
    test_R = init_R(test_list,user_dict,movie_dict)

    def measure(predictor):
        y_pred = predictor[test_R>0]
        y_true = test_R[test_R>0]
        r,c = predictor.shape
        y_pred2 = np.ones((r,c))
        y_pred2 = y_pred2*average
        y_pred2 = y_pred2[test_R>0]
        y_pred3 = np.random.rand(r,c)
        y_pred3 = y_pred3*5
        y_pred3 = y_pred3[test_R>0]

        rmse1 = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse2 = np.sqrt(mean_squared_error(y_true, y_pred2))
        rmse3 = np.sqrt(mean_squared_error(y_true, y_pred3))
        mae1 = mean_absolute_error(y_true, y_pred)
        mae2 = mean_absolute_error(y_true, y_pred2)
        return rmse1, rmse2, rmse3, mae1, mae2
    
    if base_type == 'user_based':
        model = autorec(users_num = fix_n,
                    items_num = fix_m,
                    hidden_size = hidden_size,
                    batch_size = batch_size,
                    learning_rate = learning_rate,
                    lamda_regularizer = lamda_regularizer)

        records_list = []
        for epoch in range(epoches):
            loss = model.train(train_R)
            pred_mat = model.predict_ratings(train_R)
            rmse1, rmse2, rmse3, mae1, mae2 = measure(pred_mat)
            records_list.append([rmse1, rmse2, rmse3, mae1, mae2])
        return records_list[-1]
    else:#'item_type'
        model = autorec(users_num = fix_m,
                    items_num = fix_n,
                    hidden_size = hidden_size,
                    batch_size = batch_size,
                    learning_rate = learning_rate,
                    lamda_regularizer = lamda_regularizer)

        records_list = []
        for epoch in range(epoches):
            loss = model.train(train_R.T)
            pred_mat = model.predict_ratings(train_R.T)
            rmse1, rmse2, rmse3, mae1, mae2 = measure(pred_mat.T)
            records_list.append([rmse1, rmse2, rmse3, mae1, mae2])
        return records_list[-1]


print(AutoRec(ratings_2,'user_based'))
print(AutoRec(ratings_2,'item_based'))
#[0.8229285432048727, 0.8703417809817589, 2.163051772519914, 0.6615634552483419, 0.6918135545291734]
#[0.8474361305693394, 0.869435262702052, 2.130691162688141, 0.6783440395466332, 0.7007319257234423]