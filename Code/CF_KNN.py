#*******************************Improvment_Method********************************#
#In this file, I will try some collaborative Filtering method based on the idea of similarity between viewers or movies.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

ratings_loc_1 = 'data/ratings_mini_1.csv'
ratings_1 = pd.read_csv(ratings_loc_1, index_col=None)
ratings_loc_2 = 'data/ratings_mini_2.csv'
ratings_2 = pd.read_csv(ratings_loc_2, index_col=None)
ratings_loc_3 = 'data/ratings_mini.csv'
ratings_3 = pd.read_csv(ratings_loc_3, index_col=None)
ratings_loc_4 = 'data/ratings.csv'
ratings_4 = pd.read_csv(ratings_loc_4, index_col=None)


def KNN_User(data, S, M1='pearson', M2='user_based', mci=3):
    # data: [ratings_1 or ratings_2], K: min_similarity, M1: ['pearson' or 'consine'], M2: ['user_based' or 'item_based'] 
    ratings = data
    user_dict = {}
    movie_dict = {}
    fix_n = len(ratings['userId'].unique())
    fix_m = len(ratings['movieId'].unique())
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

    def calculate_similarity(a, b, model, minimum_common_items):
        common_items = a * b > 0
        common_items_num = np.sum(common_items, axis=1)

        if model == 'pearson':
            mean_a = np.sum(a, axis=1)/(np.sum(a>0, axis=1)+1e-10)
            mean_b = np.sum(b, axis=1)/(np.sum(b>0, axis=1)+1e-10)
            aa = (a - np.reshape(mean_a, (-1,1)))*common_items
            bb = (b - np.reshape(mean_b, (-1,1)))*common_items
        else:#'cosine'
            mean_u = np.sum(b, axis=0)/(np.sum(b>0, axis=0) + 1e-10)
            aa = (a - mean_u)*common_items
            bb = (b - mean_u)*common_items

        sim = np.sum(aa*bb, axis=1)/(np.sqrt(np.sum(aa**2, axis=1))*np.sqrt(np.sum(bb**2, axis=1)) + 1e-10)
        least_common_items = common_items_num>minimum_common_items
        return sim*least_common_items

    def similarity_matrix(mat, model, minimum_common_items):
        n = mat.shape[0]
        sim_list=[]
        for u in range(n):
            a = np.tile(mat[u,:], (n,1))
            if model=='pearson':
                sim = calculate_similarity(a, mat, 'pearson', minimum_common_items)
            else:
                sim = calculate_similarity(a, mat, 'consine', minimum_common_items)
            sim_list.append(sim)
        return np.array(sim_list)
    
    if M2 == 'user_based':
        sim_mat = similarity_matrix(train_R, M1, mci)
    else:
        sim_mat = similarity_matrix(train_R.T, M1, mci)

    def prediction(train_mat, sim_mat, model, min_similarity):
        n,m = train_mat.shape

        num = np.sum(sim_mat[:,1:] > min_similarity, axis=1)#drop itself
        num_sort = np.sort(-num)
        line = int(min_similarity*len(sim_mat))
        K = -1*num_sort[line]

        if  model=='user_based':
            sim_sort = -1*np.sort(-np.array(sim_mat))[:,1:K+1]
            neighbors = np.argsort(-np.array(sim_mat))[:,1:K+1]
            common_items = train_mat[neighbors]>0 
            mean_user = np.reshape(np.sum(train_mat, axis=1)/np.sum(train_mat>0, axis=1), (-1,1))
            mat_m = train_mat - mean_user
            aa = np.sum(sim_sort[:,:,np.newaxis]*mat_m[neighbors]*common_items, axis=1)
            bb = np.sum(sim_sort[:,:,np.newaxis]*common_items, axis=1)+1e-10
            r_pred = mean_user + aa/bb
            return r_pred
        else: 
            r_pred=[]
            for u in range(n):
                u_mat = np.tile(train_mat[u], (m,1))
                rated_items_sim = (u_mat>0)*sim_mat
                sim_sort = -1*np.sort(-np.array(rated_items_sim))[:,:K]
                neighbors = np.argsort(-np.array(rated_items_sim))[:,:K]
                neighbor_ratings = np.array([u_mat[i,neighbors[i]] for i in range(m)])
                aa = np.sum(sim_sort*neighbor_ratings, axis=1)
                bb = np.sum(sim_sort, axis=1) + 1e-10 
                r_pred.append(aa/bb)

            return np.array(r_pred)
    
    r_pred = prediction(train_R, sim_mat, M2, S)
    
    def measure(predictor):
        y_pred = predictor[test_R>0]
        r,c = predictor.shape
        y_pred2 = np.ones((r,c))
        y_pred2 = y_pred2*average
        y_pred2 = y_pred2[test_R>0]
        y_pred3 = np.random.rand(r,c)
        y_pred3 = y_pred3*5
        y_pred3 = y_pred3[test_R>0]

        y_true = test_R[test_R>0]
        rmse1 = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse2 = np.sqrt(mean_squared_error(y_true, y_pred2))
        rmse3 = np.sqrt(mean_squared_error(y_true, y_pred3))
        mae1 = mean_absolute_error(y_true,y_pred)
        mae2 = mean_absolute_error(y_true,y_pred2)

        return rmse1, rmse2, rmse3, mae1, mae2
    
    perf = measure(r_pred)

    return perf

for time in range(3):
    print(time, 0.75, 'pearson', 'user_based', 5, KNN_User(ratings_2, 0.7, 'pearson', 'user_based',4))
#0 0.75 pearson user_based 5 (0.9564200892798914, 0.89829729828238, 2.1742788474904784, 0.7520286558884755, 0.7143364124799699)
#1 0.75 pearson user_based 5 (0.9319816466087896, 0.8898539240385882, 2.197438173108785, 0.7208214692971804, 0.7143505514186068)
#2 0.75 pearson user_based 5 (0.9055884955761427, 0.8627389883794174, 2.1246349022791255, 0.693008457260033, 0.6842845697049675)

for time in range(3):
    print(time, 0.75, 'consine', 'user_based', 5, KNN_User(ratings_2, 0.7, 'consine', 'user_based',4))
#0 0.75 consine user_based 5 (0.9441773185056757, 0.8563445618310557, 2.240968944847179, 0.7191406790127075, 0.6621693844848713)
#1 0.75 consine user_based 5 (0.9235111417312543, 0.8499430611414983, 2.146319519925157, 0.704983567543989, 0.671617965878028)
#2 0.75 consine user_based 5 (0.9559443155121624, 0.9169986752252068, 2.2558597707845265, 0.7359532179012259, 0.727276369120558)

for time in range(3):
    print(time, 0.6, 'pearson', 'item_based', 3, KNN_User(ratings_2, 0.6, 'pearson', 'item_based',3))
#0 0.6 pearson item_based 3 (0.8133984923847849, 0.8452768435921632, 2.004575024288081, 0.6376431403227887, 0.677968234517862)
#1 0.6 pearson item_based 3 (0.8660019836813203, 0.8969594760773479, 2.172862125008449, 0.6891598993998822, 0.7152026581204638)
#2 0.6 pearson item_based 3 (0.8141453740500467, 0.8463324591144664, 2.070976908677721, 0.6276481383545243, 0.6718069563578095)

for time in range(3):
    print(time, 0.6, 'cosine', 'item_based', 3, KNN_User(ratings_2, 0.6, 'cosine', 'item_based',3))
#0 0.6 cosine item_based 3 (0.8564632094693998, 0.8823102317690543, 2.157953050157436, 0.6722029170337962, 0.6923282118955604)
#1 0.6 cosine item_based 3 (0.811301805938326, 0.8423415615113687, 2.135924470054807, 0.6196023030021655, 0.661260250730512)
#2 0.6 cosine item_based 3 (0.8310698985300162, 0.8639303899790658, 2.148841739952474, 0.6300828515341459, 0.6860967103402772)