import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.models.baseline import Baseline
from src.analysis.models.esbm_rec import Esbm
from src.analysis.models.dc_esbm_rec import Dcesbm
from src.analysis.utilities.numba_functions import compute_log_likelihood
from src.analysis.utilities.valid_functs import generate_val_set


class TestIndividualFunctions:
    
    def setup_method(self):
        """Setup common test parameters"""
        
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
    
    # generate val set test
    @pytest.mark.parametrize("val_size, only_observed", [
        (0.1, True), (0.1, False), (0.25, True), (0.25, False), (0.5, True), (0.5, False)
    ])
    def test_generate_val_set(self, val_size, only_observed):
        Y = np.random.choice([0, 1, 2, 3, 4, 5], 
                             size=(20, 30), 
                             p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
        
        Y_train, Y_val = generate_val_set(Y, 
                                          size=val_size, 
                                          seed=42, 
                                          only_observed=only_observed)
        
        assert Y_train.shape == Y.shape
        assert len(Y_val) == int(Y_train.size*val_size)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(Y_val, list)
        
        for (i, j, v) in Y_val:
            assert Y[i, j] == v
            assert Y_train[i, j] == 0
        
        if only_observed is True:
            for (i, j, v) in Y_val:
                assert v != 0
        
        
