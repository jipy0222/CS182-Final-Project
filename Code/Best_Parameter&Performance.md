# NMF

"default" is the negative RMSE error in validation set(as a measurement for choosing best hyperparameter).

"rmse_true","rmse_avg","rmse_ran","mae_true","mae_avg" are performance in test set.

## mini1

ID:4jHWY5mv
K:8
steps:100
alpha:0.003

{"default": -0.7405463959363828, "rmse_true": 0.9589552143498027, "rmse_avg": 1.0659558300552174, "rmse_r": 1.9948201651888404, "mae_true": 0.6999583862527291, "mae_avg": 0.8521148167448325}

## mini2

ID:qOWUocwy

K:18

steps:100

alpha:0.003

{"default": -0.6605773313643065, "rmse_true": 0.8283423430725143,

"rmse_avg": 0.9109091311684795, "rmse_r": 2.1349695315270982, 

"mae_true": 0.6505762229462843, "mae_avg": 0.7337092590291818

# PMF

## mini1

ID:9x3yADLh

K:7
steps:100
alpha:0.005
beta1:0.1
beta2:0.08

{"default": -0.7100197361013615, "rmse_true": 0.8975825616051656, "rmse_avg": 0.993488440953712, "rmse_r": 1.8278959899665663, "mae_true": 0.689378567520088, "mae_avg": 0.8026315789473685}

## mini2

ID:48Jdatk7

K:13

steps:100

alpha:0.003

beta1:0.1

beta2:0.1

{"default": -0.6457912228221809, "rmse_true": 0.6836935127151379, 

"rmse_avg": 0.8102853380528096, "rmse_r": 2.2349658202874987, 

"mae_true": 0.5325073534622125, "mae_avg": 0.6599820790554608}

# KNN

## mini1

ID:TfYVD8rg

S:0.5
M1:"pearson"
M2:"user_based"
mci:3

{"default": -0.793694897846089, "rmse_true": 0.8719899066475113, "rmse_avg": 0.9375211805091772, "rmse_r": 1.938803417358715, "mae_true": 0.6679766683432201, "mae_avg": 0.7556655665566557}

S:0.5
M1:"consine"
M2:"user_based"
mci:3

{"default": -0.821391947808571, "rmse_true": 1.0561891278482352, "rmse_avg": 0.9523682045313374, "rmse_r": 1.9593987593570659, "mae_true": 0.8138147129885476, "mae_avg": 0.776185513288171}

S:0.5
M1:"pearson"
M2:"item_based"
mci:3

{"default": -1.5891930355313555, "rmse_true": 1.6879903695796112, "rmse_avg": 0.9626421881232252, "rmse_r": 1.8118301654804547, "mae_true": 1.251773963742157, "mae_avg": 0.777813965607087}

S:0.55
M1:"consine"
M2:"item_based"
mci:3

{"default": -1.6232277785472466, "rmse_true": 1.8348478593627644, "rmse_avg": 1.0233522530850658, "rmse_r": 1.8854336839035468, "mae_true": 1.373333333591053, "mae_avg": 0.8626622662266228}

## mini2

ID:LAywKeZi

S:0.45

M1:"consine"

M2:"item_based"

mci:5

{"default": -0.6335918429545762, "rmse_true": 0.7710169593914895, 

"rmse_avg": 0.9478901226752994, "rmse_r": 2.1357461830593576, 

"mae_true": 0.5797512045385851, "mae_avg": 0.7367804797241078}



S:0.45

M1:"pearson"

M2:"user_based"

mci:5

{"default": -0.6894675115709237, "rmse_true": 0.7429704468678072, 

"rmse_avg": 0.8028199599603268, "rmse_r": 2.092015820170493, 

"mae_true": 0.5908114308019372, "mae_avg": 0.6585269435377299}



S:0.45

M1:"pearson"

M2:"item_based"

mci:5

{"default": -0.7508530602835009, "rmse_true": 0.8763411737262952, 

"rmse_avg": 0.9446044350957999, "rmse_r": 2.2691669222491777, 

"mae_true": 0.6697043090494929, "mae_avg": 0.7497731645150237}



S:0.5

M1:"consine"

M2:"user_based"

mci:5

{"default": -0.7793731843499232, "rmse_true": 0.9807996480792013, 

"rmse_avg": 0.9235606662141217, "rmse_r": 2.204376041855228, 

"mae_true": 0.772703461457129, "mae_avg": 0.751595378203463}



# DL

## mini1

user_based

hidden_size:250

batch_size:128

beta:0.0001

alpha:0.01

{"default": -0.8080954900825644, "rmse_true": 0.9221749416936905, 

"rmse_avg": 0.9863419503320806, "rmse_r": 2.0036945066179395, 

"mae_true": 0.7305969055493673, "mae_avg": 0.788063806380638}



item_based

hidden_size:250

batch_size:128

beta:0.001

alpha:0.0005

{"default": -0.7460407959196453, "rmse_true": 0.8767345652781093, 

"rmse_avg": 0.9863419503320806, "rmse_r":  1.8476764851525456, 

"mae_true": 0.6959693209330241, "mae_avg": 0.788063806380638}



## mini2

user_based

hidden_size:250

batch_size:512

beta:0.001

alpha:0.005

{"default": -0.75152401539494, "rmse_true": 0.7405116927214621, 

"rmse_avg":  0.893743717036386, "rmse_r":  2.2076169940961607, 

"mae_true": 0.5877079667971116, "mae_avg": 0.7259366140709533}



item_based

hidden_size:125

batch_size:256

beta:0.00001

alpha:0.005

{"default": -0.7870337705122509, "rmse_true": 0.8195463570775467, 

"rmse_avg":  0.893743717036386, "rmse_r":  2.189054132416049, 

"mae_true": 0.6549929204837296, "mae_avg": 0.7259366140709533}



# AUTOGLUON

## mini1

[0.8950206541477348, 1.0083311185434676, 1.9118292936738492, 0.7023999283188268, 0.8025929927457941]

*** Summary of fit() ***
Estimated performance of each model:
                  model  score_val  pred_time_val  fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0   WeightedEnsemble_L2  -0.816436       0.030712  2.255615                0.000343           0.214279            2       True         11
1              LightGBM  -0.868791       0.004127  0.550269                0.004127           0.550269            1       True          4
2        KNeighborsUnif  -0.878286       0.006098  0.003604                0.006098           0.003604            1       True          1
3              CatBoost  -0.882628       0.004538  0.684898                0.004538           0.684898            1       True          6
4         LightGBMLarge  -0.889173       0.008166  0.682876                0.008166           0.682876            1       True         10
5            LightGBMXT  -0.893390       0.015758  1.484757                0.015758           1.484757            1       True          3
6       RandomForestMSE  -0.903798       0.047055  0.676047                0.047055           0.676047            1       True          5
7               XGBoost  -0.907324       0.008207  0.536790                0.008207           0.536790            1       True          9
8        KNeighborsDist  -0.915659       0.004387  0.002706                0.004387           0.002706            1       True          2
9       NeuralNetFastAI  -0.948047       0.013834  3.048254                0.013834           3.048254            1       True          8
10        ExtraTreesMSE  -0.956247       0.039016  0.506535                0.039016           0.506535            1       True          7
Number of models trained: 11
Types of models trained:
{'NNFastAiTabularModel', 'KNNModel', 'RFModel', 'CatBoostModel', 'XGBoostModel', 'WeightedEnsembleModel', 'LGBModel', 'XTModel'}
Bagging used: False 
Multi-layer stack-ensembling used: False 
Feature Metadata (Processed):
(raw dtype, special dtypes):
('category', [])                    :  1 | ['genres']
('category', ['text_as_category'])  :  1 | ['title']
('int', [])                         :  3 | ['userId', 'movieId', 'timestamp']
('int', ['binned', 'text_special']) : 11 | ['title.char_count', 'title.word_count', 'title.capital_ratio', 'title.lower_ratio', 'title.digit_ratio', ...]
Plot summary of models saved to file: AutogluonModels/ag-20220109_054736/SummaryOfModels.html
*** End of fit() summary ***

## mini2

 [0.6728123640530519, 0.8117154223705839, 2.1515192990271905, 0.5258329931155655, 0.6465984482741315]

*** Summary of fit() ***
Estimated performance of each model:
                 model  score_val  pred_time_val  fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0  WeightedEnsemble_L2  -0.700507       0.087892  7.004273                0.000313           0.186776            2       True          8
1             CatBoost  -0.716512       0.005476  1.201034                0.005476           1.201034            1       True          4
2      RandomForestMSE  -0.723017       0.050444  1.030811                0.050444           1.030811            1       True          3
3      NeuralNetFastAI  -0.756511       0.020086  4.450506                0.020086           4.450506            1       True          6
4        ExtraTreesMSE  -0.791729       0.066432  0.721056                0.066432           0.721056            1       True          5
5       KNeighborsUnif  -0.835061       0.005948  0.005888                0.005948           0.005888            1       True          1
6       KNeighborsDist  -0.836037       0.005295  0.003400                0.005295           0.003400            1       True          2
7              XGBoost  -3.110721       0.006277  0.131747                0.006277           0.131747            1       True          7
Number of models trained: 8
Types of models trained:
{'NNFastAiTabularModel', 'KNNModel', 'RFModel', 'CatBoostModel', 'XGBoostModel', 'WeightedEnsembleModel', 'XTModel'}
Bagging used: False 
Multi-layer stack-ensembling used: False 
Feature Metadata (Processed):
(raw dtype, special dtypes):
('category', [])                    :  1 | ['genres']
('category', ['text_as_category'])  :  1 | ['title']
('int', [])                         :  3 | ['userId', 'movieId', 'timestamp']
('int', ['binned', 'text_special']) : 15 | ['title.char_count', 'title.word_count', 'title.capital_ratio', 'title.lower_ratio', 'title.digit_ratio', ...]
Plot summary of models saved to file: AutogluonModels/ag-20220109_054750/SummaryOfModels.html
*** End of fit() summary ***
