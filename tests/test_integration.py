from matplotlib.pylab import seed
import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.models.baseline import Baseline
from src.analysis.models.esbm_rec import esbm
from src.analysis.models.dc_esbm_rec import dcesbm
from src.analysis.utilities.valid_functs import * 


class TestIntegration:
    
    def setup_method(self):
        """Setup common test parameters"""
        
        Y = np.array(([
                [2, 1, 0, 0, 0, 2, 2, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 2, 0, 0],
                [4, 2, 0, 1, 2, 3, 7, 1, 1, 2],
                [3, 3, 4, 1, 3, 5, 4, 3, 0, 0],
                [4, 3, 1, 5, 4, 1, 3, 3, 2, 2],
                [2, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                [1, 2, 0, 0, 0, 2, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 2, 1, 2, 0, 0],
                [3, 8, 1, 0, 2, 2, 4, 4, 3, 1],
                [6, 4, 0, 1, 4, 8, 5, 5, 3, 5]]))
        
        self.valid_params = {
            'num_items': 10,
            'num_users': 10,
            'prior_a': 1,
            'prior_b': 1,
            'Y': Y,
            'epsilon': 1e-6,
            'seed': 1,
            'verbose_users': False,
            'verbose_items': False,
            'device': 'cpu'
        }
        
        self.runs_params = {
            'n_users': 30,
            'n_items': 30,
            'num_clusters_users': 4,
            'num_clusters_items': 4,
            'n_runs': 1,
            'n_iters': 100,
            'burn_in': 25,
            'thinning': 2,
            'k': 5,
            'n_runs': 1,
        }
        
    @pytest.mark.parametrize("scheme_type", ['DP', 'PY', 'GN'])
    def test_equality_of_llk_init(self, scheme_type):
        modelbaseline = Baseline(scheme_type=scheme_type, **self.valid_params)
        modelesbm = esbm(scheme_type=scheme_type, **self.valid_params)
        modeldcesbm = dcesbm(scheme_type=scheme_type, **self.valid_params)

        baseline_llk = modelbaseline.compute_log_likelihood()
        esbm_llk = modelesbm.compute_log_likelihood()
        dcesbm_llk = modeldcesbm.compute_log_likelihood()
        assert np.isclose(baseline_llk, esbm_llk)
        assert np.isclose(baseline_llk, dcesbm_llk)
    
    # tests for function working
    @pytest.mark.parametrize("scheme_type", ['DP', 'PY', 'GN'])
    def test_better_than_baseline(self, scheme_type):
        modelbaseline = Baseline(scheme_type=scheme_type, **self.valid_params)
        modelesbm = esbm(scheme_type=scheme_type, **self.valid_params)
        modeldcesbm = dcesbm(scheme_type=scheme_type, **self.valid_params)

        outbaseline = modelbaseline.gibbs_train(100)
        outesbm = modelesbm.gibbs_train(100)
        outdcesbm = modeldcesbm.gibbs_train(100)
        
        baselinellk_final = outbaseline[0][-1]
        esbmllk_final = outesbm[0][-1]
        dcesbmllk_final = outdcesbm[0][-1]
        assert esbmllk_final > baselinellk_final
        assert dcesbmllk_final > baselinellk_final

    @pytest.mark.parametrize("true_mod", ['dcesbm'])
    def test_multiple_runs(self, true_mod):
        
        if true_mod == 'dcesbm':
            true_mod = dcesbm
        else:
            true_mod = esbm
        
        n_users = 30
        n_items = 30
        num_clusters_users = 4
        num_clusters_items = 4
        
        seed = 1
        n_iters = 100
        burn_in = 25
        thinning = 2
        k = 5
        n_runs = 1
        
        params_init = {
            'num_users': n_users,
            'num_items': n_items,
            'bar_h_users': num_clusters_users,
            'bar_h_items': num_clusters_items,
            'item_clustering': 'random',
            'user_clustering': 'random',
            'degree_param_users': 5,
            'degree_param_items': 5,
            'scheme_type': 'DM',
            'seed': 42,
            'sigma': -0.9
        }
        
        params_dp = {'scheme_type': 'DP'}
        params_py = {'scheme_type': 'PY',}
        params_gn = {'scheme_type': 'GN',}
        params_dp_cov = {'scheme_type':'DP'}
        params_py_cov = {'scheme_type':'PY'}
        params_gn_cov = {'scheme_type':'GN'}
        
        params_list = [params_dp, params_py, params_gn,
                       params_dp_cov, params_py_cov, params_gn_cov,
                        params_dp, params_py, params_gn, 
                        params_dp_cov, params_py_cov, params_gn_cov]
        
        model_list = [dcesbm, dcesbm, dcesbm,
                       dcesbm, dcesbm, dcesbm,
                       esbm, esbm, esbm, esbm, esbm, esbm]
        
        model_names = ['dcesbm_dp', 'dcesbm_py', 'dcesbm_gn',
                       'dcesbm_dp_cov', 'dcesbm_py_cov', 'dcesbm_gn_cov',
                       'esbm_dp', 'esbm_py', 'esbm_gn',
                       'esbm_dp_cov', 'esbm_py_cov', 'esbm_gn_cov']
        
        cov_places_users = [3,4,5]#, 9, 10, 11]
        cov_places_items = [3,4,5]#, 9, 10, 11]
        
        out = multiple_runs(
            true_mod=true_mod, 
            params_init=params_init, 
            num_users=n_users, 
            num_items=n_items, 
            num_clusters_users=num_clusters_users, 
            num_clusters_items=num_clusters_items, 
            n_runs=n_runs, 
            n_iters=n_iters,
            params_list=params_list, 
            model_list=model_list, 
            model_names=model_names, 
            cov_places_users=cov_places_users, 
            cov_places_items=cov_places_items, 
            k=k, 
            print_intermid=True, 
            verbose=1, 
            burn_in=burn_in, 
            thinning=thinning, 
            seed=seed)
        
        assert out is not None
        
