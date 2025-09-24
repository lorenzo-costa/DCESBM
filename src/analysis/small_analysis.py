import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.plotting_functions import plot_heatmap
from models.esbm_rec import Esbm
from models.dc_esbm_rec import Dcesbm
from utilities.valid_functs import multiple_runs
import yaml
import seaborn as sns
from utilities.valid_functs import generate_val_set, validate_models


# load and and set parameters
with open("src/analysis/config_sim.yaml", "r") as f:
    config = yaml.safe_load(f)

n_users = 100
n_items = 100
n_iters = 100
burn_in = 50
thinning = 3
k = 10
n_runs = 2
seed = config["run_settings"]["seed"]

params_baseline = config["params_baseline"]
params_dp = config["params_variants"]["dp"]
params_py = config["params_variants"]["py"]
params_gn = config["params_variants"]["gn"]
params_dp_cov = config["params_variants"]["dp_cov"]
params_py_cov = config["params_variants"]["py_cov"]
params_gn_cov = config["params_variants"]["gn_cov"]
params_init = config["params_init"]

params_init['num_users'] = n_users
params_init['num_items'] = n_items

cov_places_users = [3,4,5]
cov_places_items = [3,4,5]

model_list = [Dcesbm, Dcesbm, Dcesbm, Dcesbm, Dcesbm, Dcesbm, 
              Esbm, Esbm, Esbm, Esbm, Esbm, Esbm]
params_list = [params_dp, params_py, params_gn, 
               params_dp_cov, params_py_cov, params_gn_cov,
               params_dp, params_py, params_gn, 
               params_dp_cov, params_py_cov, params_gn_cov]

model_names = ['dc_DP', 'dc_PY', 'dc_GN', 
               'dc_DP_cov', 'dc_PY_cov', 'dc_GN_cov',
               'esbm_DP', 'esbm_PY', 'esbm_GN', 
               'esbm_DP_cov', 'esbm_PY_cov', 'esbm_GN_cov']

# run simulations 
print('\nStarting simulations with', n_runs, 'runs of', n_iters, 'iterations each.')

out = multiple_runs(true_mod=Dcesbm, 
                    params_init=params_init, 
                    num_users=n_users, 
                    num_items=n_items,  
                    n_runs=n_runs, 
                    n_iters=n_iters,
                    params_list=params_list, 
                    model_list=model_list, 
                    model_names=model_names, 
                    cov_places_users=cov_places_users, 
                    cov_places_items=cov_places_items, 
                    k=k, 
                    verbose=1, 
                    burn_in=burn_in, 
                    thinning=thinning, 
                    seed=seed) 

print('\nSimulations completed! Saving results...')
# extract and save results
names_list=out[0]
mse_list=out[1]
mae_list=out[2]
precision_list=out[3]
recall_list=out[4]
vi_users_list=out[5]
vi_items_list=out[6]
models_list_out=out[7]

dc_mean_dp_mse = np.mean(mse_list[0::12])
dc_mean_py_mse = np.mean(mse_list[1::12])
dc_mean_gn_mse = np.mean(mse_list[2::12])
dc_mean_dp_cov_mse = np.mean(mse_list[3::12])
dc_mean_py_cov_mse = np.mean(mse_list[4::12])
dc_mean_gn_cov_mse = np.mean(mse_list[5::12])

dc_mean_dp_mae = np.mean(mae_list[0::12])
dc_mean_py_mae = np.mean(mae_list[1::12])
dc_mean_gn_mae = np.mean(mae_list[2::12])
dc_mean_dp_cov_mae = np.mean(mae_list[3::12])
dc_mean_py_cov_mae = np.mean(mae_list[4::12])
dc_mean_gn_cov_mae = np.mean(mae_list[5::12])

dc_mean_dp_prec = np.mean(precision_list[0::12])
dc_mean_py_prec = np.mean(precision_list[1::12])
dc_mean_gn_prec = np.mean(precision_list[2::12])
dc_mean_dp_cov_prec = np.mean(precision_list[3::12])
dc_mean_py_cov_prec = np.mean(precision_list[4::12])
dc_mean_gn_cov_prec = np.mean(precision_list[5::12])

dc_mean_dp_rec = np.mean(recall_list[0::12])
dc_mean_py_rec = np.mean(recall_list[1::12])
dc_mean_gn_rec = np.mean(recall_list[2::12])
dc_mean_dp_cov_rec = np.mean(recall_list[3::12])
dc_mean_py_cov_rec = np.mean(recall_list[4::12])
dc_mean_gn_cov_rec = np.mean(recall_list[5::12])

dc_mean_dp_vi_users = np.mean(vi_users_list[0::12])
dc_mean_py_vi_users = np.mean(vi_users_list[1::12])
dc_mean_gn_vi_users = np.mean(vi_users_list[2::12])
dc_mean_dp_cov_vi_users = np.mean(vi_users_list[3::12])
dc_mean_py_cov_vi_users = np.mean(vi_users_list[4::12])
dc_mean_gn_cov_vi_users = np.mean(vi_users_list[5::12])

dc_mean_dp_vi_items = np.mean(vi_items_list[0::12])
dc_mean_py_vi_items = np.mean(vi_items_list[1::12])
dc_mean_gn_vi_items = np.mean(vi_items_list[2::12])
dc_mean_dp_cov_vi_items = np.mean(vi_items_list[3::12])
dc_mean_py_cov_vi_items = np.mean(vi_items_list[4::12])
dc_mean_gn_cov_vi_items = np.mean(vi_items_list[5::12])

esbm_mean_dp_mse = np.mean(mse_list[6::12])
esbm_mean_py_mse = np.mean(mse_list[7::12])
esbm_mean_gn_mse = np.mean(mse_list[8::12])
esbm_mean_dp_cov_mse = np.mean(mse_list[9::12])
esbm_mean_py_cov_mse = np.mean(mse_list[10::12])
esbm_mean_gn_cov_mse = np.mean(mse_list[11::12])    

esbm_mean_dp_mae = np.mean(mae_list[6::12])
esbm_mean_py_mae = np.mean(mae_list[7::12])
esbm_mean_gn_mae = np.mean(mae_list[8::12])
esbm_mean_dp_cov_mae = np.mean(mae_list[9::12])
esbm_mean_py_cov_mae = np.mean(mae_list[10::12])
esbm_mean_gn_cov_mae = np.mean(mae_list[11::12])

esbm_mean_dp_prec = np.mean(precision_list[6::12])
esbm_mean_py_prec = np.mean(precision_list[7::12])
esbm_mean_gn_prec = np.mean(precision_list[8::12])
esbm_mean_dp_cov_prec = np.mean(precision_list[9::12])
esbm_mean_py_cov_prec = np.mean(precision_list[10::12])
esbm_mean_gn_cov_prec = np.mean(precision_list[11::12])

esbm_mean_dp_rec = np.mean(recall_list[6::12])
esbm_mean_py_rec = np.mean(recall_list[7::12])
esbm_mean_gn_rec = np.mean(recall_list[8::12])
esbm_mean_dp_cov_rec = np.mean(recall_list[9::12])
esbm_mean_py_cov_rec = np.mean(recall_list[10::12])
esbm_mean_gn_cov_rec = np.mean(recall_list[11::12])

esbm_mean_dp_vi_users = np.mean(vi_users_list[6::12])
esbm_mean_py_vi_users = np.mean(vi_users_list[7::12])
esbm_mean_gn_vi_users = np.mean(vi_users_list[8::12])
esbm_mean_dp_cov_vi_users = np.mean(vi_users_list[9::12])
esbm_mean_py_cov_vi_users = np.mean(vi_users_list[10::12])
esbm_mean_gn_cov_vi_users = np.mean(vi_users_list[11::12])

esbm_mean_dp_vi_items = np.mean(vi_items_list[6::12])
esbm_mean_py_vi_items = np.mean(vi_items_list[7::12])
esbm_mean_gn_vi_items = np.mean(vi_items_list[8::12])
esbm_mean_dp_cov_vi_items = np.mean(vi_items_list[9::12])
esbm_mean_py_cov_vi_items = np.mean(vi_items_list[10::12])
esbm_mean_gn_cov_vi_items = np.mean(vi_items_list[11::12])


# make output table
output_table = pd.DataFrame()

output_table['Model'] = ['dc_DP', 'dc_PY', 'dc_GN', 
                         'dc_DP_cov', 'dc_PY_cov', 'dc_GN_cov',
                         'esbm_DP', 'esbm_PY', 'esbm_GN', 
                         'esbm_DP_cov', 'esbm_PY_cov', 'esbm_GN_cov']

output_table['MAE'] = [dc_mean_dp_mae, dc_mean_py_mae, dc_mean_gn_mae, 
                       dc_mean_dp_cov_mae, dc_mean_py_cov_mae, dc_mean_gn_cov_mae,
                       esbm_mean_dp_mae, esbm_mean_py_mae, esbm_mean_gn_mae, 
                       esbm_mean_dp_cov_mae, esbm_mean_py_cov_mae, esbm_mean_gn_cov_mae]

output_table['MSE'] = [dc_mean_dp_mse, dc_mean_py_mse, dc_mean_gn_mse, 
                       dc_mean_dp_cov_mse, dc_mean_py_cov_mse, dc_mean_gn_cov_mse,
                       esbm_mean_dp_mse, esbm_mean_py_mse, esbm_mean_gn_mse, 
                       esbm_mean_dp_cov_mse, esbm_mean_py_cov_mse, esbm_mean_gn_cov_mse]

output_table['Precision'] = [dc_mean_dp_prec, dc_mean_py_prec, dc_mean_gn_prec, 
                             dc_mean_dp_cov_prec, dc_mean_py_cov_prec, dc_mean_gn_cov_prec,
                             esbm_mean_dp_prec, esbm_mean_py_prec, esbm_mean_gn_prec, 
                             esbm_mean_dp_cov_prec, esbm_mean_py_cov_prec, esbm_mean_gn_cov_prec]

output_table['Recall'] = [dc_mean_dp_rec, dc_mean_py_rec, dc_mean_gn_rec, 
                          dc_mean_dp_cov_rec, dc_mean_py_cov_rec, dc_mean_gn_cov_rec,
                          esbm_mean_dp_rec, esbm_mean_py_rec, esbm_mean_gn_rec, 
                          esbm_mean_dp_cov_rec, esbm_mean_py_cov_rec, esbm_mean_gn_cov_rec]

output_table['VI_users'] = [dc_mean_dp_vi_users, dc_mean_py_vi_users, dc_mean_gn_vi_users, 
                            dc_mean_dp_cov_vi_users, dc_mean_py_cov_vi_users, dc_mean_gn_cov_vi_users,
                            esbm_mean_dp_vi_users, esbm_mean_py_vi_users, esbm_mean_gn_vi_users, 
                            esbm_mean_dp_cov_vi_users, esbm_mean_py_cov_vi_users, esbm_mean_gn_cov_vi_users]

output_table['VI_items'] = [dc_mean_dp_vi_items, dc_mean_py_vi_items, dc_mean_gn_vi_items, 
                            dc_mean_dp_cov_vi_items, dc_mean_py_cov_vi_items, dc_mean_gn_cov_vi_items,
                            esbm_mean_dp_vi_items, esbm_mean_py_vi_items, esbm_mean_gn_vi_items, 
                            esbm_mean_dp_cov_vi_items, esbm_mean_py_cov_vi_items, esbm_mean_gn_cov_vi_items]


output_table.to_csv('results/small/results_simulations.csv', index=False)


# extract and save llk plots
model_dp_dc = models_list_out[0]
model_py_dc = models_list_out[1]
model_gn_dc = models_list_out[2]
model_dp_cov_dc = models_list_out[3]
model_py_cov_dc = models_list_out[4]
model_gn_cov_dc = models_list_out[5]
model_dp_esbm = models_list_out[6]
model_py_esbm = models_list_out[7]
model_gn_esbm = models_list_out[8]
model_dp_cov_esbm = models_list_out[9]
model_py_cov_esbm = models_list_out[10]
model_gn_cov_esbm = models_list_out[11]

llk_dp_dc = model_dp_dc.train_llk
llk_py_dc = model_py_dc.train_llk
llk_gn_dc = model_gn_dc.train_llk
llk_dp_dc_cov = model_dp_cov_dc.train_llk
llk_py_dc_cov = model_py_cov_dc.train_llk
llk_gn_dc_cov = model_gn_cov_dc.train_llk
llk_dp_esbm = model_dp_esbm.train_llk
llk_py_esbm = model_py_esbm.train_llk
llk_gn_esbm = model_gn_esbm.train_llk
llk_dp_esbm_cov = model_dp_cov_esbm.train_llk
llk_py_esbm_cov = model_py_cov_esbm.train_llk
llk_gn_esbm_cov = model_gn_cov_esbm.train_llk

plot_heatmap(model_dp_dc, save_path='results/small/heatmap_dp_dc.png')
plot_heatmap(model_dp_cov_dc, save_path='results/small/heatmap_dp_cov_dc.png')
plot_heatmap(model_dp_esbm, save_path='results/small/heatmap_dp_esbm.png')
plot_heatmap(model_dp_cov_esbm, save_path='results/small/heatmap_dp_cov_esbm.png')

# plot by group type
groups = [
    {'data': [llk_dp_dc, llk_py_dc, llk_gn_dc], 
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for DC models',
     'path': 'results/small/llk_dc.png'},
    {'data': [llk_dp_dc_cov, llk_py_dc_cov, llk_gn_dc_cov],
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for DC models with covariates',
     'path': 'results/small/llk_dc_cov.png'},
    {'data': [llk_dp_esbm, llk_py_esbm, llk_gn_esbm],
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for ESBM models',
     'path': 'results/small/llk_esbm.png'},
    {'data': [llk_dp_esbm_cov, llk_py_esbm_cov, llk_gn_esbm_cov],
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for ESBM models with covariates',
     'path': 'results/small/llk_esbm_cov.png'}]

for group in groups:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for data, label in zip(group['data'], group['labels']):
        ax.plot(data[2:], label=label)
    
    ax.legend()
    plt.title(group['title'])
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.tight_layout()
    plt.savefig(group['path'], dpi=300, bbox_inches='tight')
    


###############################################################################
#-------------------------------------------------------------------------------
# book analysis part
#-------------------------------------------------------------------------------

print('\n\nStarting book analysis...')
# load dataset
print('\nLoading dataset...')
dataset_clean = pd.read_csv('data/processed/dataset_clean.csv')

# load settings
with open("src/analysis/config_books.yaml", "r") as f:
    config = yaml.safe_load(f)
    
n_users = 100
n_items = 100
n_cl_u = 5
n_cl_i = 5
params_dp = config["params_variants"]["dp"]
params_py = config["params_variants"]["py"]
params_gn = config["params_variants"]["gn"]
params_dp_cov = config["params_variants"]["dp_cov"]
params_py_cov = config["params_variants"]["py_cov"]
params_gn_cov = config["params_variants"]["gn_cov"]
n_iters = 500
burn_in = 10
thinning = 1
k = 10
seed = 42


# data preparation
print('\nPreparing data...')
# Create user-item matrix and take subset
matrix_form = dataset_clean.pivot_table(index='user_id', columns='book_id', values='rating', fill_value=0).astype(int)
matrix_form = matrix_form.to_numpy()
matrix_small = matrix_form[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]][:, np.flip(np.argsort(matrix_form.sum(axis=0)))[:n_items]].copy()

fig, ax = plt.subplots(figsize=(10, 6))
plt.title('User-Book Heatmap')
sns.heatmap(matrix_small, ax=ax)
ax.set_xlabel('Books')
ax.set_ylabel('Users')
plt.tight_layout()
plt.savefig('results/small/user-book_heatmap.png')

# Create user covariates
cov_biography = dataset_clean['biography'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_fiction = dataset_clean['fiction'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_history = dataset_clean['history'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_classic = dataset_clean['classic'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_romance = dataset_clean['romance'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_items = [
    ('cat_biography', cov_biography[:n_items]),
    ('cat_fiction', cov_fiction[:n_items]),
    ('cat_history', cov_history[:n_items]),
    ('cat_classic', cov_classic[:n_items]),
    ('cat_romance', cov_romance[:n_items])]

# train-test split
Y_train, y_val = generate_val_set(matrix_small, size=0.2, seed=42, only_observed=False)


# training
# define models
model_list = [Esbm, Esbm, Esbm, Esbm, Esbm, Esbm, Dcesbm, Dcesbm, Dcesbm, Dcesbm, Dcesbm, Dcesbm]
params_list = [params_dp, params_py, params_gn, params_dp_cov, params_py_cov, params_gn_cov, params_dp, params_py, params_gn, params_dp_cov, params_py_cov, params_gn_cov]  
model_names = ['esbm_DP', 'esbm_PY', 'esbm_GN', 'esbm_DP_COV', 'esbm_PY_COV', 'esbm_GN_COV', 'dcesbm_DP', 'dcesbm_PY', 'dcesbm_GN', 'dcesbm_DP_COV', 'dcesbm_PY_COV', 'dcesbm_GN_COV']

# Validate models
print('\n\nStarting model validation...')
out_models = validate_models(Y_train, 
                             y_val, 
                             model_list, 
                             params_list, 
                             n_iters=n_iters, 
                             burn_in=burn_in, 
                             k=k,
                             verbose=1, 
                             thinning=thinning, 
                             model_names=model_names, 
                             seed=seed)

print('\n\nModel validation completed! Saving results...')

# extract models
esbm_dp = out_models[0]
esbm_py = out_models[1]
esbm_gn = out_models[2]
esbm_dp_cov = out_models[3]
esbm_py_cov = out_models[4]
esbm_gn_cov = out_models[5]
dcesbm_dp = out_models[6]
dcesbm_py = out_models[7]
dcesbm_gn = out_models[8]
dcesbm_dp_cov = out_models[9]
dcesbm_py_cov = out_models[10]
dcesbm_gn_cov = out_models[11]

llk_esbm_dp = esbm_dp.train_llk
llk_esbm_py = esbm_py.train_llk
llk_esbm_gn = esbm_gn.train_llk
llk_esbm_dp_cov = esbm_dp_cov.train_llk
llk_esbm_py_cov = esbm_py_cov.train_llk
llk_esbm_gn_cov = esbm_gn_cov.train_llk
llk_dcesbm_dp = dcesbm_dp.train_llk
llk_dcesbm_py = dcesbm_py.train_llk
llk_dcesbm_gn = dcesbm_gn.train_llk
llk_dcesbm_dp_cov = dcesbm_dp_cov.train_llk
llk_dcesbm_py_cov = dcesbm_py_cov.train_llk
llk_dcesbm_gn_cov = dcesbm_gn_cov.train_llk


# plot and save llk plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(llk_esbm_dp, label='esbm_DP')
ax1.plot(llk_esbm_py, label='esbm_PY')
ax1.plot(llk_esbm_gn, label='esbm_GN')
ax1.legend()
plt.title('ESBM Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/small/llk_esbm.png')

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(llk_esbm_dp_cov, label='esbm_DP_COV')
ax2.plot(llk_esbm_py_cov, label='esbm_PY_COV')
ax2.plot(llk_esbm_gn_cov, label='esbm_GN_COV')
ax2.legend()
plt.title('ESBM with Covariates Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/small/llk_esbm_cov.png')

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(llk_dcesbm_dp, label='dcesbm_DP')
ax3.plot(llk_dcesbm_py, label='dcesbm_PY')
ax3.plot(llk_dcesbm_gn, label='dcesbm_GN')
ax3.legend()
plt.title('DCESBM Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/small/llk_dcesbm.png')

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(llk_dcesbm_dp_cov, label='dcesbm_DP_COV')
ax4.plot(llk_dcesbm_py_cov, label='dcesbm_PY_COV')
ax4.plot(llk_dcesbm_gn_cov, label='dcesbm_GN_COV')
ax4.legend()
plt.title('DCESBM with Covariates Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/small/llk_dcesbm_cov.png')


# extract and save val metrics
mae_esbm_dp = esbm_dp.mae
mse_esbm_dp = esbm_dp.mse
precision_esbm_dp = esbm_dp.precision
recall_esbm_dp = esbm_dp.recall

mae_esbm_py = esbm_py.mae
mse_esbm_py = esbm_py.mse
precision_esbm_py = esbm_py.precision
recall_esbm_py = esbm_py.recall

mae_esbm_gn = esbm_gn.mae
mse_esbm_gn = esbm_gn.mse
precision_esbm_gn = esbm_gn.precision
recall_esbm_gn = esbm_gn.recall

mae_esbm_dp_cov = esbm_dp_cov.mae
mse_esbm_dp_cov = esbm_dp_cov.mse
precision_esbm_dp_cov = esbm_dp_cov.precision
recall_esbm_dp_cov = esbm_dp_cov.recall

mae_esbm_py_cov = esbm_py_cov.mae
mse_esbm_py_cov = esbm_py_cov.mse
precision_esbm_py_cov = esbm_py_cov.precision
recall_esbm_py_cov = esbm_py_cov.recall

mae_esbm_gn_cov = esbm_gn_cov.mae
mse_esbm_gn_cov = esbm_gn_cov.mse
precision_esbm_gn_cov = esbm_gn_cov.precision
recall_esbm_gn_cov = esbm_gn_cov.recall

mae_dcesbm_dp = dcesbm_dp.mae
mse_dcesbm_dp = dcesbm_dp.mse
precision_dcesbm_dp = dcesbm_dp.precision
recall_dcesbm_dp = dcesbm_dp.recall

mae_dcesbm_py = dcesbm_py.mae
mse_dcesbm_py = dcesbm_py.mse
precision_dcesbm_py = dcesbm_py.precision
recall_dcesbm_py = dcesbm_py.recall

mae_dcesbm_gn = dcesbm_gn.mae
mse_dcesbm_gn = dcesbm_gn.mse
precision_dcesbm_gn = dcesbm_gn.precision
recall_dcesbm_gn = dcesbm_gn.recall

mae_dcesbm_dp_cov = dcesbm_dp_cov.mae
mse_dcesbm_dp_cov = dcesbm_dp_cov.mse
precision_dcesbm_dp_cov = dcesbm_dp_cov.precision
recall_dcesbm_dp_cov = dcesbm_dp_cov.recall

mae_dcesbm_py_cov = dcesbm_py_cov.mae
mse_dcesbm_py_cov = dcesbm_py_cov.mse
precision_dcesbm_py_cov = dcesbm_py_cov.precision
recall_dcesbm_py_cov = dcesbm_py_cov.recall

mae_dcesbm_gn_cov = dcesbm_gn_cov.mae
mse_dcesbm_gn_cov = dcesbm_gn_cov.mse
precision_dcesbm_gn_cov = dcesbm_gn_cov.precision
recall_dcesbm_gn_cov = dcesbm_gn_cov.recall

output_table = pd.DataFrame()
output_table['Model'] = ['esbm_DP', 'esbm_PY', 'esbm_GN', 'esbm_DP_COV', 'esbm_PY_COV', 'esbm_GN_COV',
                        'dcesbm_DP', 'dcesbm_PY', 'dcesbm_GN', 'dcesbm_DP_COV', 'dcesbm_PY_COV', 'dcesbm_GN_COV']

output_table['MAE'] = [mae_esbm_dp, mae_esbm_py, mae_esbm_gn, 
                       mae_esbm_dp_cov, mae_esbm_py_cov, mae_esbm_gn_cov, 
                       mae_dcesbm_dp, mae_dcesbm_py, mae_dcesbm_gn, 
                       mae_dcesbm_dp_cov, mae_dcesbm_py_cov, mae_dcesbm_gn_cov]
output_table['MSE'] = [mse_esbm_dp, mse_esbm_py, mse_esbm_gn, 
                       mse_esbm_dp_cov, mse_esbm_py_cov, mse_esbm_gn_cov,
                       mse_dcesbm_dp, mse_dcesbm_py, mse_dcesbm_gn,
                       mse_dcesbm_dp_cov, mse_dcesbm_py_cov, mse_dcesbm_gn_cov]
output_table['Precision'] = [precision_esbm_dp, precision_esbm_py, precision_esbm_gn,
                             precision_esbm_dp_cov, precision_esbm_py_cov, precision_esbm_gn_cov,
                             precision_dcesbm_dp, precision_dcesbm_py, precision_dcesbm_gn,
                             precision_dcesbm_dp_cov, precision_dcesbm_py_cov, precision_dcesbm_gn_cov]
output_table['Recall'] = [recall_esbm_dp, recall_esbm_py, recall_esbm_gn,
                          recall_esbm_dp_cov, recall_esbm_py_cov, recall_esbm_gn_cov,
                          recall_dcesbm_dp, recall_dcesbm_py, recall_dcesbm_gn,
                          recall_dcesbm_dp_cov, recall_dcesbm_py_cov, recall_dcesbm_gn_cov]

output_table.to_csv('results/small/results_books.csv', index=False)

