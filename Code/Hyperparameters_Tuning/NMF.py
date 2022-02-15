#**************************Baseline_Method**************************#
# NOTICE: We will use CF_as_Matrix_Completion(NMF) as the baseline method.
from nni.trial import report_intermediate_result
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
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


def NMF(data, K, steps, alpha):
    # data:[ratings_1 or ratings_2 or ratings_3 or raitngs_4], K: new matrix dimension,
    # steps: max_iterations for SGD, alpha: learning rate
    ratings = data
    fix_n = len(ratings['userId'].unique())
    fix_m = len(ratings['movieId'].unique())
    user_dict = {}
    movie_dict = {}
    average = train_list['rating'].mean()

    def init_R_W_H(source, user_dict, movie_dict):
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
        user_pre = np.random.normal(0, 0.1, (n, K))
        movie_weight = np.random.normal(0, 0.1, (m, K))
        return R, user_pre, movie_weight

    train_R, user_pre, movie_weight = init_R_W_H(
        train_list, user_dict, movie_dict)
    valid_true, temp1, temp2 = init_R_W_H(valid_list, user_dict, movie_dict)
    test_true, temp3, temp4 = init_R_W_H(test_list, user_dict, movie_dict)

    def matrix_factorization(P, Q, steps, alpha):
        records_list = []
        for step in range(steps):
            los = 0.0
            for index, row in train_list.iterrows():
                u = user_dict[int(row['userId'])]
                i = movie_dict[int(row['movieId'])]
                r = float(row['rating'])
                P[u], Q[i], ls = update(P[u], Q[i], r=r, learning_rate=alpha)
                los += ls
            pred_mat = prediction(P, Q)
            records_list = valid_measure(pred_mat)+measure(pred_mat)
            if los < 0.00001:
                break
        return P, Q, np.array(records_list)

    def update(p, q, r, learning_rate=0.001):
        error = r - np.dot(p, q.T)
        p = p + learning_rate*error*q
        q = q + learning_rate*error*p
        loss = 0.5 * error**2
        return p, q, loss

    def prediction(P, Q):
        N, K = P.shape
        M, K = Q.shape
        rating_list = []
        for u in range(N):
            u_rating = np.sum(P[u, :]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred

    def valid_measure(predictor):
        y_pred = predictor[valid_true > 0]
        y_true = valid_true[valid_true > 0]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return [-rmse]

    def measure(predictor):
        y_pred = predictor[test_true > 0]
        y_true = test_true[test_true > 0]
        r, c = predictor.shape
        y_pred2 = np.ones((r, c))
        y_pred2 = y_pred2*average
        y_pred2 = y_pred2[test_true > 0]
        y_pred3 = np.random.rand(r, c)
        y_pred3 = y_pred3*5
        y_pred3 = y_pred3[test_true > 0]

        rmse1 = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse2 = np.sqrt(mean_squared_error(y_true, y_pred2))
        rmse3 = np.sqrt(mean_squared_error(y_true, y_pred3))
        mae1 = mean_absolute_error(y_true, y_pred)
        mae2 = mean_absolute_error(y_true, y_pred2)
        return [rmse1, rmse2, rmse3, mae1, mae2]

    user_pre, movie_weight, perf = matrix_factorization(
        user_pre, movie_weight, steps, alpha)

    return perf


def main_nni(args):
    ret = NMF(ratings_2, K=args['K'], steps=args['steps'], alpha=args['alpha'])
    ret = dict(zip(['default', 'rmse_true','rmse_avg', 'rmse_r',
                    'mae_true', 'mae_avg'], ret))
    nni.report_final_result(ret)


if __name__ == '__main__':
    args = nni.get_next_parameter()
    main_nni(args)

# ID:qOWUocwy
# K:18
# steps:100
# alpha:0.003
# {"default": -0.6605773313643065, "rmse_true": 0.8283423430725143,
# "rmse_avg": 0.9109091311684795, "rmse_r": 2.1349695315270982, 
# "mae_true": 0.6505762229462843, "mae_avg": 0.7337092590291818}
