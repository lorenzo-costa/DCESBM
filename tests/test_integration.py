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
    
    
         
