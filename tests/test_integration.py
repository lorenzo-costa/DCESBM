import pytest
import sys
from pathlib import Path
import numpy as np
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.models.baseline import Baseline
from src.analysis.models.esbm_rec import Esbm
from src.analysis.models.dc_esbm_rec import Dcesbm
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
        
        n_users_runs, n_items_runs = 50, 50
        num_clusters_users_runs, num_clusters_items_runs = 4, 4
        
        self.run_params_init = {
            'num_users': n_users_runs,
            'num_items': n_items_runs,
            'bar_h_users': num_clusters_users_runs,
            'bar_h_items': num_clusters_items_runs,
            'item_clustering': 'random',
            'user_clustering': 'random',
            'degree_param_users': 10,
            'degree_param_items': 10,
            'scheme_type': 'DM',
            'seed': 42,
            'sigma': -0.9
        }
        
        self.runs_params = {
            'params_init': self.run_params_init,
            'num_users': n_users_runs,
            'num_items': n_items_runs,
            'n_runs': 1,
            'n_iters': 1500,
            'burn_in': 250,
            'thinning': 2,
            'k': 5,
            'n_runs': 1,
        }
        
    @pytest.mark.parametrize("scheme_type", ['DP', 'PY', 'GN'])
    def test_equality_of_llk_init(self, scheme_type):
        modelbaseline = Baseline(scheme_type=scheme_type, **self.valid_params)
        modelesbm = Esbm(scheme_type=scheme_type, **self.valid_params)
        modeldcesbm = Dcesbm(scheme_type=scheme_type, **self.valid_params)

        baseline_llk = modelbaseline.compute_log_likelihood()
        esbm_llk = modelesbm.compute_log_likelihood()
        dcesbm_llk = modeldcesbm.compute_log_likelihood()
        assert np.isclose(baseline_llk, esbm_llk)
        assert np.isclose(baseline_llk, dcesbm_llk)
    
    # tests for function working
    @pytest.mark.parametrize("scheme_type", ['DP', 'PY', 'GN'])
    def test_better_than_baseline(self, scheme_type):
        modelbaseline = Baseline(scheme_type=scheme_type, **self.valid_params)
        modelesbm = Esbm(scheme_type=scheme_type, **self.valid_params)
        modeldcesbm = Dcesbm(scheme_type=scheme_type, **self.valid_params)

        outbaseline = modelbaseline.gibbs_train(100)
        outesbm = modelesbm.gibbs_train(100)
        outdcesbm = modeldcesbm.gibbs_train(100)
        
        baselinellk_final = outbaseline[0][-1]
        esbmllk_final = outesbm[0][-1]
        dcesbmllk_final = outdcesbm[0][-1]
        assert esbmllk_final > baselinellk_final
        assert dcesbmllk_final > baselinellk_final
        
    @pytest.mark.parametrize("true_model", ['dcesbm', 'esbm'])
    def test_multiple_runs_nocov(self, true_model):
        if true_model == 'dcesbm':
            true_mod = Dcesbm
        else:
            true_mod = Esbm
        
        params_dp = {'scheme_type': 'DP', 'degree_param_users': 5, 'degree_param_items': 5,}
        params_py = {'scheme_type': 'PY', 'degree_param_users': 5, 'degree_param_items': 5,}
        params_gn = {'scheme_type': 'GN', 'degree_param_users': 5, 'degree_param_items': 5,}

        params_list = [params_dp, params_py, params_gn, 
                       params_dp, params_py, params_gn]

        model_list = [Dcesbm, Dcesbm, Dcesbm, Esbm, Esbm, Esbm]

        model_names = ['dcesbm_dp', 'dcesbm_py', 'dcesbm_gn', 'esbm_dp', 'esbm_py', 'esbm_gn']

        out = multiple_runs(
            true_mod=true_mod, 
            params_list=params_list, 
            model_list=model_list, 
            model_names=model_names,  
            print_intermid=True, 
            verbose=0,
            **self.runs_params)
        
        assert out is not None
        
        vi_users_list=out[5]
        vi_items_list=out[6]
        
        n_runs = self.runs_params['n_runs']

        vi_users_dcesbm_dp = np.mean(vi_users_list[0::n_runs])
        vi_users_dcesbm_py = np.mean(vi_users_list[1::n_runs])
        vi_users_dcesbm_gn = np.mean(vi_users_list[2::n_runs])
        vi_users_esbm_dp = np.mean(vi_users_list[3::n_runs])
        vi_users_esbm_py = np.mean(vi_users_list[4::n_runs])
        vi_users_esbm_gn = np.mean(vi_users_list[5::n_runs])
        
        vi_items_dcesbm_dp = np.mean(vi_items_list[0::n_runs])
        vi_items_dcesbm_py = np.mean(vi_items_list[1::n_runs])
        vi_items_dcesbm_gn = np.mean(vi_items_list[2::n_runs])
        vi_items_esbm_dp = np.mean(vi_items_list[3::n_runs])
        vi_items_esbm_py = np.mean(vi_items_list[4::n_runs])
        vi_items_esbm_gn = np.mean(vi_items_list[5::n_runs])

        # if true model is dcesbm, dcesbm should recover true structure better
        if true_model == 'dcesbm':
            # if they close to each other, just warn
            if not vi_users_dcesbm_dp < vi_users_esbm_dp:
                assert np.abs(vi_users_dcesbm_dp-vi_users_esbm_dp) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_users_dcesbm_dp={vi_users_dcesbm_dp}, vi_users_esbm_dp={vi_users_esbm_dp}")
            if not vi_users_dcesbm_py < vi_users_esbm_py:
                assert np.abs(vi_users_dcesbm_py-vi_users_esbm_py) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_users_dcesbm_py={vi_users_dcesbm_py}, vi_users_esbm_py={vi_users_esbm_py}")
            if not vi_users_dcesbm_gn < vi_users_esbm_gn:
                assert np.abs(vi_users_dcesbm_gn-vi_users_esbm_gn) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_users_dcesbm_gn={vi_users_dcesbm_gn}, vi_users_esbm_gn={vi_users_esbm_gn}")
            if not vi_items_dcesbm_dp < vi_items_esbm_dp:
                assert np.abs(vi_items_dcesbm_dp-vi_items_esbm_dp) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_items_dcesbm_dp={vi_items_dcesbm_dp}, vi_items_esbm_dp={vi_items_esbm_dp}")
            if not vi_items_dcesbm_py < vi_items_esbm_py:
                assert np.abs(vi_items_dcesbm_py-vi_items_esbm_py) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_items_dcesbm_py={vi_items_dcesbm_py}, vi_items_esbm_py={vi_items_esbm_py}")
            if not vi_items_dcesbm_gn < vi_items_esbm_gn:
                assert np.abs(vi_items_dcesbm_gn-vi_items_esbm_gn) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_items_dcesbm_gn={vi_items_dcesbm_gn}, vi_items_esbm_gn={vi_items_esbm_gn}")
        else:
            if not vi_users_esbm_dp < vi_users_dcesbm_dp:
                assert np.abs(vi_users_esbm_dp-vi_users_dcesbm_dp) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_users_esbm_dp={vi_users_esbm_dp}, vi_users_dcesbm_dp={vi_users_dcesbm_dp}")
            if not vi_users_esbm_py < vi_users_dcesbm_py:
                assert np.abs(vi_users_esbm_py-vi_users_dcesbm_py) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_users_esbm_py={vi_users_esbm_py}, vi_users_dcesbm_py={vi_users_dcesbm_py}")
            if not vi_users_esbm_gn < vi_users_dcesbm_gn:
                assert np.abs(vi_users_esbm_gn-vi_users_dcesbm_gn) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_users_esbm_gn={vi_users_esbm_gn}, vi_users_dcesbm_gn={vi_users_dcesbm_gn}")
            if not vi_items_esbm_dp < vi_items_dcesbm_dp:
                assert np.abs(vi_items_esbm_dp-vi_items_dcesbm_dp) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_items_esbm_dp={vi_items_esbm_dp}, vi_items_dcesbm_dp={vi_items_dcesbm_dp}")
            if not vi_items_esbm_py < vi_items_dcesbm_py:
                assert np.abs(vi_items_esbm_py-vi_items_dcesbm_py) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_items_esbm_py={vi_items_esbm_py}, vi_items_dcesbm_py={vi_items_dcesbm_py}")
            if not vi_items_esbm_gn < vi_items_dcesbm_gn:
                assert np.abs(vi_items_esbm_gn-vi_items_dcesbm_gn) < 1e-1
                warnings.warn(f"Warning: true model barely close to alternative\n"
                              f" vi_items_esbm_gn={vi_items_esbm_gn}, vi_items_dcesbm_gn={vi_items_dcesbm_gn}")

    @pytest.mark.parametrize("true_model", ['dcesbm', 'esbm'])
    def test_multiple_runs_cov(self, true_model):
        if true_model == 'dcesbm':
            true_mod = Dcesbm
        else:
            true_mod = Esbm

        params_dp = {'scheme_type': 'DP', 'degree_param_users': 5, 'degree_param_items': 5,}
        params_py = {'scheme_type': 'PY', 'degree_param_users': 5, 'degree_param_items': 5,}
        params_gn = {'scheme_type': 'GN', 'degree_param_users': 5, 'degree_param_items': 5,}

        params_list = [params_dp, params_py, params_gn, 
                       params_dp, params_py, params_gn]

        model_list = [Dcesbm, Dcesbm, Dcesbm, Esbm, Esbm, Esbm]

        model_names = ['dcesbm_dp', 'dcesbm_py', 'dcesbm_gn', 'esbm_dp', 'esbm_py', 'esbm_gn']
        
        cov_places_users = [0, 1, 2, 3, 4, 5,]
        cov_places_items = [0, 1, 2, 3, 4, 5,]

        out = multiple_runs(
            true_mod=true_mod, 
            params_list=params_list, 
            model_list=model_list, 
            model_names=model_names,  
            print_intermid=True, 
            cov_places_users=cov_places_users,
            cov_places_items=cov_places_items,
            verbose=0,
            **self.runs_params)
        
        assert out is not None
        
        vi_users_list=out[5]
        vi_items_list=out[6]
        
        n_runs = self.runs_params['n_runs']

        vi_users_dcesbm_dp = np.mean(vi_users_list[0::n_runs])
        vi_users_dcesbm_py = np.mean(vi_users_list[1::n_runs])
        vi_users_dcesbm_gn = np.mean(vi_users_list[2::n_runs])
        vi_users_esbm_dp = np.mean(vi_users_list[3::n_runs])
        vi_users_esbm_py = np.mean(vi_users_list[4::n_runs])
        vi_users_esbm_gn = np.mean(vi_users_list[5::n_runs])
        
        vi_items_dcesbm_dp = np.mean(vi_items_list[0::n_runs])
        vi_items_dcesbm_py = np.mean(vi_items_list[1::n_runs])
        vi_items_dcesbm_gn = np.mean(vi_items_list[2::n_runs])
        vi_items_esbm_dp = np.mean(vi_items_list[3::n_runs])
        vi_items_esbm_py = np.mean(vi_items_list[4::n_runs])
        vi_items_esbm_gn = np.mean(vi_items_list[5::n_runs])

        # if true model is dcesbm, dcesbm should recover true structure better
        if true_model == 'dcesbm':
            # if they close to each other, just warn
            if not vi_users_dcesbm_dp < vi_users_esbm_dp:
                assert np.abs(vi_users_dcesbm_dp-vi_users_esbm_dp) < 1e-1
                warnings.warn(f"Warning: true model (dcesbm) barely close to alternative\n"
                              f" vi_users_dcesbm_dp={vi_users_dcesbm_dp}, vi_users_esbm_dp={vi_users_esbm_dp}")
            if not vi_users_dcesbm_py < vi_users_esbm_py:
                assert np.abs(vi_users_dcesbm_py-vi_users_esbm_py) < 1e-1
                warnings.warn(f"Warning: true model (dcesbm) barely close to alternative\n"
                              f" vi_users_dcesbm_py={vi_users_dcesbm_py}, vi_users_esbm_py={vi_users_esbm_py}")
            if not vi_users_dcesbm_gn < vi_users_esbm_gn:
                assert np.abs(vi_users_dcesbm_gn-vi_users_esbm_gn) < 1e-1
                warnings.warn(f"Warning: true model (dcesbm) barely close to alternative\n"
                              f" vi_users_dcesbm_gn={vi_users_dcesbm_gn}, vi_users_esbm_gn={vi_users_esbm_gn}")
            if not vi_items_dcesbm_dp < vi_items_esbm_dp:
                assert np.abs(vi_items_dcesbm_dp-vi_items_esbm_dp) < 1e-1
                warnings.warn(f"Warning: true model (dcesbm) barely close to alternative\n"
                              f" vi_items_dcesbm_dp={vi_items_dcesbm_dp}, vi_items_esbm_dp={vi_items_esbm_dp}")
            if not vi_items_dcesbm_py < vi_items_esbm_py:
                assert np.abs(vi_items_dcesbm_py-vi_items_esbm_py) < 1e-1
                warnings.warn(f"Warning: true model (dcesbm) barely close to alternative\n"
                              f" vi_items_dcesbm_py={vi_items_dcesbm_py}, vi_items_esbm_py={vi_items_esbm_py}")
            if not vi_items_dcesbm_gn < vi_items_esbm_gn:
                assert np.abs(vi_items_dcesbm_gn-vi_items_esbm_gn) < 1e-1
                warnings.warn(f"Warning: true model (dcesbm) barely close to alternative\n"
                              f" vi_items_dcesbm_gn={vi_items_dcesbm_gn}, vi_items_esbm_gn={vi_items_esbm_gn}")
        else:
            if not vi_users_esbm_dp < vi_users_dcesbm_dp:
                assert np.abs(vi_users_esbm_dp-vi_users_dcesbm_dp) < 1e-1
                warnings.warn(f"Warning: true model (esbm) barely close to alternative\n"
                              f" vi_users_esbm_dp={vi_users_esbm_dp}, vi_users_dcesbm_dp={vi_users_dcesbm_dp}")
            if not vi_users_esbm_py < vi_users_dcesbm_py:
                assert np.abs(vi_users_esbm_py-vi_users_dcesbm_py) < 1e-1
                warnings.warn(f"Warning: true model (esbm) barely close to alternative\n"
                              f" vi_users_esbm_py={vi_users_esbm_py}, vi_users_dcesbm_py={vi_users_dcesbm_py}")
            if not vi_users_esbm_gn < vi_users_dcesbm_gn:
                assert np.abs(vi_users_esbm_gn-vi_users_dcesbm_gn) < 1e-1
                warnings.warn(f"Warning: true model (esbm) barely close to alternative\n"
                              f" vi_users_esbm_gn={vi_users_esbm_gn}, vi_users_dcesbm_gn={vi_users_dcesbm_gn}")
            if not vi_items_esbm_dp < vi_items_dcesbm_dp:
                assert np.abs(vi_items_esbm_dp-vi_items_dcesbm_dp) < 1e-1
                warnings.warn(f"Warning: true model (esbm) barely close to alternative\n"
                              f" vi_items_esbm_dp={vi_items_esbm_dp}, vi_items_dcesbm_dp={vi_items_dcesbm_dp}")
            if not vi_items_esbm_py < vi_items_dcesbm_py:
                assert np.abs(vi_items_esbm_py-vi_items_dcesbm_py) < 1e-1
                warnings.warn(f"Warning: true model (esbm) barely close to alternative\n"
                              f" vi_items_esbm_py={vi_items_esbm_py}, vi_items_dcesbm_py={vi_items_dcesbm_py}")
            if not vi_items_esbm_gn < vi_items_dcesbm_gn:
                assert np.abs(vi_items_esbm_gn-vi_items_dcesbm_gn) < 1e-1
                warnings.warn(f"Warning: true model (esbm) barely close to alternative\n"
                              f" vi_items_esbm_gn={vi_items_esbm_gn}, vi_items_dcesbm_gn={vi_items_dcesbm_gn}")

