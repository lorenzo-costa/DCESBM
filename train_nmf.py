import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.samplers import GridSampler
import joblib
import warnings
import datetime
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic, NMF
from surprise import accuracy
from hpfrec import HPF
from sklearn.model_selection import train_test_split

Y_train = np.load('code/Y_train_books.npy')

full_df = []
u_ids = []
i_ids = []
rui = []

for u in range(Y_train.shape[0]):
    for i in range(Y_train.shape[1]):
        u_ids.append(u)
        i_ids.append(i)
        rui.append(Y_train[u, i])

df_full = pd.DataFrame({'UserId': u_ids, 'ItemId': i_ids, 'Count': rui})

train_df, val_df = train_test_split(df_full, test_size=0.2, random_state=42)

reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_df[['UserId', 'ItemId', 'Count']], reader)
trainset = trainset.build_full_trainset()

reader = Reader(rating_scale=(1, 5))
valset = Dataset.load_from_df(val_df[['UserId', 'ItemId', 'Count']], reader)
valset = valset.build_full_trainset().build_testset()

params = {
  'n_epochs':100,
  'random_state':42,
  'biased':False}

def objective(trial):
  
  params['n_factors'] = trial.suggest_int('n_factors', 2, 30)
  params['reg_pu'] = trial.suggest_float('reg_pu', 0.00001, 1, log=True)
  params['reg_qi'] = trial.suggest_float('reg_qi', 0.00001, 1, log=True)
  # params['reg_bu'] = trial.suggest_float('reg_bu', 0.001, 1, log=True)
  # params['reg_bi'] = trial.suggest_float('reg_bi', 0.001, 1, log=True)
  # params['lr_bu'] = trial.suggest_float('lr_bu', 0.0001, 0.01, log=True)
  # params['lr_bi'] = trial.suggest_float('lr_bi', 0.0001, 0.01, log=True)
    
  model = NMF(**params)
  model.fit(trainset)
  
  preds = model.test(valset)
  
  return accuracy.mae(preds)

warnings.filterwarnings('ignore')
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=123))
study.optimize(objective, n_trials=150)

best_params = {k: study.best_trial.params[k] for k in study.best_trial.params.keys()}

print()
print()
fig = optuna.visualization.plot_param_importances(study)
fig.show()

print(best_params)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value, best_params))