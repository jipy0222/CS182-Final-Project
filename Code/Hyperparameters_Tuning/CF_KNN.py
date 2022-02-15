#*******************************Improvment_Method********************************#
# In this file, I will try some collaborative Filtering method based on the idea of similarity between viewers or movies.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import nni

global train_list
global valid_list
global test_list

ratings_loc_1 = 'data/ratings_mini_1.csv'  # 750
ratings_1 = pd.read_csv(ratings_loc_1, index_col=None)
ratings_loc_2 = 'data/ratings_mini_2.csv'  # 2500
ratings_2 = pd.read_csv(ratings_loc_2, index_col=None)
ratings_loc_3 = 'data/ratings_mini.csv'  # 125
ratings_3 = pd.read_csv(ratings_loc_3, index_col=None)
ratings_loc_4 = 'data/ratings.csv'
ratings_4 = pd.read_csv(ratings_loc_4, index_col=None)

train_list, test_list = train_test_split(ratings_1, test_size=0.1)
train_list, valid_list = train_test_split(train_list, test_size=(1/9))


def KNN_User(data, S, M1='pearson', M2='user_based', mci=3):
    # data: [ratings_1 or ratings_2], S: min_similarity, M1: ['pearson' or 'consine'], M2: ['user_based' or 'item_based']
    ratings = data
    user_dict = {}
    movie_dict = {}
    fix_n = len(ratings['userId'].unique())
    fix_m = len(ratings['movieId'].unique())
    average = train_list['rating'].mean()

    def init_R(source, user_dict, movie_dict):
        countnum_user = -1
        countnum_movie = -1
        n = max(len(source['userId'].unique()), fix_n)
        m = max(len(source['movieId'].unique()), fix_m)
        R = np.zeros((n, m), dtype=float)
        for index, item in source.iterrows():
            if int(item['userId']) not in user_dict.keys():
                countnum_user += 1
                user_dict[int(item['userId'])] = countnum_user
                if int(item['movieId']) not in movie_dict.keys():
                    countnum_movie += 1
                    movie_dict[int(item['movieId'])] = countnum_movie
                    R[user_dict[int(item['userId'])]][movie_dict[int(
                        item['movieId'])]] = item['rating']
                else:
                    R[user_dict[int(item['userId'])]][movie_dict[int(
                        item['movieId'])]] = item['rating']
            else:
                if int(item['movieId']) not in movie_dict.keys():
                    countnum_movie += 1
                    movie_dict[int(item['movieId'])] = countnum_movie
                    R[user_dict[int(item['userId'])]][movie_dict[int(
                        item['movieId'])]] = item['rating']
                else:
                    R[user_dict[int(item['userId'])]][movie_dict[int(
                        item['movieId'])]] = item['rating']
        return R

    train_R = init_R(train_list, user_dict, movie_dict)
    valid_R = init_R(valid_list, user_dict, movie_dict)
    test_R = init_R(test_list, user_dict, movie_dict)

    def calculate_similarity(a, b, model, minimum_common_items):
        common_items = a * b > 0
        common_items_num = np.sum(common_items, axis=1)

        if model == 'pearson':
            mean_a = np.sum(a, axis=1)/(np.sum(a > 0, axis=1)+1e-10)
            mean_b = np.sum(b, axis=1)/(np.sum(b > 0, axis=1)+1e-10)
            aa = (a - np.reshape(mean_a, (-1, 1)))*common_items
            bb = (b - np.reshape(mean_b, (-1, 1)))*common_items
        else:  # 'cosine'
            mean_u = np.sum(b, axis=0)/(np.sum(b > 0, axis=0) + 1e-10)
            aa = (a - mean_u)*common_items
            bb = (b - mean_u)*common_items

        sim = np.sum(aa*bb, axis=1)/(np.sqrt(np.sum(aa**2, axis=1))
                                     * np.sqrt(np.sum(bb**2, axis=1)) + 1e-10)
        least_common_items = common_items_num > minimum_common_items
        return sim*least_common_items

    def similarity_matrix(mat, model, minimum_common_items):
        n = mat.shape[0]
        sim_list = []
        for u in range(n):
            a = np.tile(mat[u, :], (n, 1))
            if model == 'pearson':
                sim = calculate_similarity(
                    a, mat, 'pearson', minimum_common_items)
            else:
                sim = calculate_similarity(
                    a, mat, 'consine', minimum_common_items)
            sim_list.append(sim)
        return np.array(sim_list)

    if M2 == 'user_based':
        sim_mat = similarity_matrix(train_R, M1, mci)
    else:
        sim_mat = similarity_matrix(train_R.T, M1, mci)

    def prediction(train_mat, sim_mat, model, min_similarity):
        n, m = train_mat.shape

        num = np.sum(sim_mat[:, 1:] > min_similarity, axis=1)  # drop itself
        num_sort = np.sort(-num)
        line = int(min_similarity*len(sim_mat))
        K = -1*num_sort[line]

        if model == 'user_based':
            sim_sort = -1*np.sort(-np.array(sim_mat))[:, 1:K+1]
            neighbors = np.argsort(-np.array(sim_mat))[:, 1:K+1]
            common_items = train_mat[neighbors] > 0
            mean_user = np.reshape(
                np.sum(train_mat, axis=1)/np.sum(train_mat > 0, axis=1), (-1, 1))
            mat_m = train_mat - mean_user
            aa = np.sum(sim_sort[:, :, np.newaxis] *
                        mat_m[neighbors]*common_items, axis=1)
            bb = np.sum(sim_sort[:, :, np.newaxis]*common_items, axis=1)+1e-10
            r_pred = mean_user + aa/bb
            return r_pred
        else:
            r_pred = []
            for u in range(n):
                u_mat = np.tile(train_mat[u], (m, 1))
                rated_items_sim = (u_mat > 0)*sim_mat
                sim_sort = -1*np.sort(-np.array(rated_items_sim))[:, :K]
                neighbors = np.argsort(-np.array(rated_items_sim))[:, :K]
                neighbor_ratings = np.array(
                    [u_mat[i, neighbors[i]] for i in range(m)])
                aa = np.sum(sim_sort*neighbor_ratings, axis=1)
                bb = np.sum(sim_sort, axis=1) + 1e-10
                r_pred.append(aa/bb)

            return np.array(r_pred)

    r_pred = prediction(train_R, sim_mat, M2, S)

    def valid_measure(predictor):
        y_pred = predictor[valid_R > 0]
        y_true = valid_R[valid_R > 0]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return [-rmse]

    def measure(predictor):
        y_pred = predictor[test_R > 0]
        r, c = predictor.shape
        y_pred2 = np.ones((r, c))
        y_pred2 = y_pred2*average
        y_pred2 = y_pred2[test_R > 0]
        y_pred3 = np.random.rand(r, c)
        y_pred3 = y_pred3*5
        y_pred3 = y_pred3[test_R > 0]

        y_true = test_R[test_R > 0]
        rmse1 = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse2 = np.sqrt(mean_squared_error(y_true, y_pred2))
        rmse3 = np.sqrt(mean_squared_error(y_true, y_pred3))
        mae1 = mean_absolute_error(y_true, y_pred)
        mae2 = mean_absolute_error(y_true, y_pred2)

        return [rmse1, rmse2, rmse3, mae1, mae2]

    perf = valid_measure(r_pred)+measure(r_pred)

    return perf


def main_nni(args):
    ret = KNN_User(
        ratings_2,  S=args['S'], M1=args['M1'], M2=args['M2'], mci=args['mci'])
    ret = dict(zip(['default', 'rmse_true', 'rmse_avg', 'rmse_r',
                    'mae_true', 'mae_avg'], ret))
    nni.report_final_result(ret)


if __name__ == '__main__':
    params = nni.get_next_parameter()
    main_nni(params)

# ID:LAywKeZi
# S:0.45
# M1:"consine"
# M2:"item_based"
# mci:5
# {"default": -0.6335918429545762, "rmse_true": 0.7710169593914895, 
# "rmse_avg": 0.9478901226752994, "rmse_r": 2.1357461830593576, 
# "mae_true": 0.5797512045385851, "mae_avg": 0.7367804797241078}

# S:0.45
# M1:"pearson"
# M2:"user_based"
# mci:5
# {"default": -0.6894675115709237, "rmse_true": 0.7429704468678072, 
# "rmse_avg": 0.8028199599603268, "rmse_r": 2.092015820170493, 
# "mae_true": 0.5908114308019372, "mae_avg": 0.6585269435377299}

# S:0.45
# M1:"pearson"
# M2:"item_based"
# mci:5
# {"default": -0.7508530602835009, "rmse_true": 0.8763411737262952, 
# "rmse_avg": 0.9446044350957999, "rmse_r": 2.2691669222491777, 
# "mae_true": 0.6697043090494929, "mae_avg": 0.7497731645150237}

# S:0.5
# M1:"consine"
# M2:"user_based"
# mci:5
# {"default": -0.7793731843499232, "rmse_true": 0.9807996480792013, 
# "rmse_avg": 0.9235606662141217, "rmse_r": 2.204376041855228, 
# "mae_true": 0.772703461457129, "mae_avg": 0.751595378203463}

# c&i >> p&u >> p&i >> c&u in the rank 100
