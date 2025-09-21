import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.models.baseline import Baseline
from src.analysis.models.esbm_rec import Esbm
from src.analysis.models.dc_esbm_rec import Dcesbm
from src.analysis.utilities.numba_functions import compute_log_likelihood


class TestIndividualFunctions:
    
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
            'epsilon': 1e-6,
            'seed': 1,
            'verbose_users': False,
            'verbose_items': False,
            'device': 'cpu'
        }
        
        self.cov_users = [('cov1_cat', np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])),
                          ('cov2_cat', np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))]
        self.cov_items = [('cov1_cat', np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])),
                          ('cov2_cat', np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))]

    # gibbs step and train tests
    @pytest.mark.parametrize("model, scheme_type",[
        (Baseline, 'DP'), (Baseline, 'PY'), (Baseline, 'GN'),
        (Esbm, 'DP'), (Esbm, 'PY'), (Esbm, 'GN'),
        (Dcesbm, 'DP'), (Dcesbm, 'PY'), (Dcesbm, 'GN')])
    def test_gibbs_step(self, model, scheme_type):
        """test gibbs_step runs without error"""
        model = model(scheme_type=scheme_type, **self.valid_params)
        model.gibbs_step()
        assert True
    
    @pytest.mark.parametrize("model, scheme_type",[
        (Baseline, 'DP'), (Baseline, 'PY'), (Baseline, 'GN'),
        (Esbm, 'DP'), (Esbm, 'PY'), (Esbm, 'GN'),
        (Dcesbm, 'DP'), (Dcesbm, 'PY'), (Dcesbm, 'GN')])
    def test_gibbs_train(self, model, scheme_type):
        """test gibbs_train runs without error"""
        model = model(scheme_type=scheme_type, **self.valid_params)
        model.gibbs_train(100)
        assert True
    
    # gibbs with covariates and train tests
    @pytest.mark.parametrize("model, scheme_type",[
        (Baseline, 'DP'), (Baseline, 'PY'), (Baseline, 'GN'),
        (Esbm, 'DP'), (Esbm, 'PY'), (Esbm, 'GN'),
        (Dcesbm, 'DP'), (Dcesbm, 'PY'), (Dcesbm, 'GN')])
    def test_gibbs_step_cov(self, model, scheme_type):
        """test gibbs_step runs without error"""
        model = model(scheme_type=scheme_type, 
                      cov_users=self.cov_users, 
                      cov_items=self.cov_items,
                      **self.valid_params)
        model.gibbs_step()
        assert True
    
    @pytest.mark.parametrize("model, scheme_type",[
        (Baseline, 'DP'), (Baseline, 'PY'), (Baseline, 'GN'),
        (Esbm, 'DP'), (Esbm, 'PY'), (Esbm, 'GN'),
        (Dcesbm, 'DP'), (Dcesbm, 'PY'), (Dcesbm, 'GN')])
    def test_gibbs_train_cov(self, model, scheme_type):
        """test gibbs_train runs without error"""
        model = model(scheme_type=scheme_type, 
                      cov_users=self.cov_users, 
                      cov_items=self.cov_items,
                      **self.valid_params)
        model.gibbs_train(100)
        assert True
        
    # log likelihood tests
    @pytest.mark.parametrize(("model, degree_corrected, scheme_type"),
                             [(Baseline, False, 'DP'), (Esbm, False, 'DP'), (Dcesbm, True, 'DP')])
    def test_compute_log_likelihood(self, model, degree_corrected, scheme_type):
        """tests that self.compute_log_likelihood() returns the same value as compute_log_likelihood()"""
        mm = model(scheme_type=scheme_type, **self.valid_params)
        if degree_corrected:
            llk = compute_log_likelihood(
                nh = mm.frequencies_users, 
                nk = mm.frequencies_items, 
                a = mm.prior_a, 
                b = mm.prior_b, 
                eps = mm.epsilon, 
                mhk=mm._compute_mhk(), 
                user_clustering=mm.user_clustering, 
                item_clustering=mm.item_clustering,
                degree_param_users=1,
                degree_param_items=1,
                dg_u=mm.degree_users, 
                dg_i=mm.degree_items, 
                dg_cl_i=mm.degree_clusters_items, 
                dg_cl_u=mm.degree_clusters_users,
                degree_corrected=True)
        else:
            llk = compute_log_likelihood(
                nh = mm.frequencies_users, 
                nk = mm.frequencies_items, 
                a = mm.prior_a, 
                b = mm.prior_b, 
                eps = mm.epsilon, 
                mhk=mm._compute_mhk(), 
                user_clustering=mm.user_clustering, 
                item_clustering=mm.item_clustering,
                degree_param_users=1,
                degree_param_items=1,
                dg_u=np.zeros(mm.num_users), 
                dg_i=np.zeros(mm.num_items), 
                dg_cl_i=np.zeros(mm.num_clusters_items), 
                dg_cl_u=np.zeros(mm.num_clusters_users),
                degree_corrected=False)
        
        assert np.isclose(llk, mm.compute_log_likelihood())
    
    # degree correction tests
    @pytest.mark.parametrize("scheme_type", ['DP', 'PY', 'GN'])
    def test_degree_correction_to_infinity(self, scheme_type):
        """tests that when degree correction parameter goes to infinity dc 
        model is the same as esbm"""
        params_dc = self.valid_params.copy()
        params_dc['degree_param_users'] = 1e15
        params_dc['degree_param_items'] = 1e15
        mm_dc = Dcesbm(scheme_type=scheme_type, **params_dc)
        mm = Esbm(scheme_type=scheme_type, **self.valid_params)

        llk_dc = mm_dc.compute_log_likelihood()
        llk = mm.compute_log_likelihood()
        
        assert np.isclose(llk_dc, llk)
    
    