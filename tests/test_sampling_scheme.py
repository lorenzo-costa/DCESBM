import sys
sys.path.append("src/analysis")
from utilities.numba_functions import sampling_scheme
import numpy as np

class TestSamplingScheme:
    def setup_method(self):
        self.params_dm = {
            'V': 10,
            'H': 2,
            'frequencies': np.array([5, 5]), # sums to V, len is H 
            'bar_h': 10,
            'scheme_type': 'DM',
            'scheme_param':4,
            'sigma':-1,
            'gamma':1
        }
        
        self.params_dp = {
            'V': 10,
            'H': 2,
            'frequencies': np.array([5, 5]),
            'bar_h': 10,
            'scheme_type': 'DP',
            'scheme_param':1,
            'sigma':0,
            'gamma':1
        }
        
        self.params_py = {
            'V': 10,
            'H': 2,
            'frequencies': np.array([5, 5]),
            'bar_h': 10,
            'scheme_type': 'PY',
            'scheme_param':1,
            'sigma':0.5,
            'gamma':1
        }
        
        self.params_gn = {
            'V': 10,
            'H': 2,
            'frequencies': np.array([5, 5]),
            'bar_h': 10,
            'scheme_type': 'GN',
            'scheme_param':0.5,
            'sigma':-1,
            'gamma':0.5
        }
        
        self.eps = 1e-6
        

    def test_DM_scheme_no_new_cluster(self):
        """"Test that DM scheme does not create new cluster when h_bar=H"""
        params_updated = self.params_dm.copy()
        params_updated['bar_h'] = params_updated['H']
        
        out = sampling_scheme(**params_updated)
        assert np.all(out == (params_updated['frequencies']-params_updated['sigma']))
        assert (sum(out/sum(out))-1) < self.eps

    def test_DM_scheme_new_cluster(self):
        """"Test that DM scheme creates new cluster when h_bar>H"""
        params_updated = self.params_dm.copy()
        
        out = sampling_scheme(**params_updated)
        expected = np.append(params_updated['frequencies'] - params_updated['sigma'], 
                             -params_updated['sigma'] * (params_updated['bar_h'] - params_updated['H']))
        assert np.all(out == expected)
        assert (sum(out/sum(out))-1) < self.eps
    
    def test_dp(self):
        params_updated = self.params_dp.copy()
        out = sampling_scheme(**params_updated)
        expected = np.append(params_updated['frequencies'], params_updated['scheme_param'])
        assert np.all(out == expected)
        assert (sum(out/sum(out))-1) < self.eps
    
    def test_py(self):
        params_updated = self.params_py.copy()
        out = sampling_scheme(**params_updated)
        
        expected = np.append(params_updated['frequencies'] - params_updated['sigma'], 
                             params_updated['scheme_param'] + params_updated['sigma'] * params_updated['H'])
        assert np.all(out == expected)
        assert (sum(out/sum(out))-1) < self.eps

    def test_gn(self):
        params_updated = self.params_gn.copy()
        out = sampling_scheme(**params_updated)
        
        expected = np.append(
            (params_updated['frequencies']+1) * (params_updated['V'] - params_updated['H'] + params_updated['gamma']),
            params_updated['H']* (params_updated['H'] - params_updated['gamma'])
            )
        assert np.all(out == expected)
        assert (sum(out/sum(out))-1) < self.eps
    
    def test_py_equal_dp(self):
        """Test that PY with sigma=0 is equivalent to DP"""
        params_py = self.params_py.copy()
        params_py['sigma']=0
        params_dp = self.params_dp.copy()
        
        out_py = sampling_scheme(**params_py)
        out_dp = sampling_scheme(**params_dp)
        
        assert np.all(out_py == out_dp)
        
        
    
    
