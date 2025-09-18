import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.analysis.utilities.numba_functions import sampling_scheme
import numpy as np
import pytest



class TestSamplingScheme:
    # base set
    params_dm = {
        'V': 10, 'H': 2, 'bar_h': 10, 'scheme_type': 'DM',
        'scheme_param': 4, 'sigma': -1, 'gamma': 1
    }

    params_dp = {
        'V': 10, 'H': 2, 'bar_h': 10, 'scheme_type': 'DP',
        'scheme_param': 1, 'sigma': 0, 'gamma': 1
    }
    
    params_py = {
        'V': 10, 'H': 2, 'bar_h': 10, 'scheme_type': 'PY',
        'scheme_param': 1, 'sigma': 0.5, 'gamma': 1
    }
    
    params_gn = {
        'V': 10, 'H': 2, 'bar_h': 10, 'scheme_type': 'GN',
        'scheme_param': 0.5, 'sigma': -1, 'gamma': 0.5
    }
    
    eps = 1e-6


    # dm should not make new clusters
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5)
        ]
    )
    def test_DM_scheme_no_new_cluster(self, V, frequencies, H):
        """Test that DM scheme does not create new cluster when bar_h=H."""
        params_updated = self.params_dm.copy()
        params_updated['frequencies'] = frequencies
        params_updated['H'] = H
        params_updated['bar_h'] = H 
        params_updated['V'] = V
        
        out = sampling_scheme(**params_updated)
        assert np.all(out == (frequencies - params_updated['sigma']))
        assert (sum(out/sum(out)) - 1) < self.eps


    # test dm behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H, bar_h",
        [
            (10, np.array([5, 5]), 2, 10),
            (10, np.array([2, 4, 4]), 3, 5),
        ]
    )
    def test_DM_scheme_new_cluster(self, V, frequencies, H, bar_h):
        """Test that DM scheme creates new cluster when bar_h>H."""
        params_updated = self.params_dm.copy()
        params_updated['frequencies'] = frequencies
        params_updated['H'] = H
        params_updated['bar_h'] = bar_h
        params_updated['V'] = V
        
        out = sampling_scheme(**params_updated)
        expected = np.append(
            frequencies - params_updated['sigma'],
            -params_updated['sigma'] * (bar_h - H)
        )
        assert np.allclose(out, expected)
        assert (sum(out/sum(out)) - 1) < self.eps


    # test dp behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5)
        ]
    )
    def test_dp(self, V, frequencies, H):
        params_updated = self.params_dp.copy()
        params_updated['frequencies'] = frequencies
        params_updated['H'] = H
        params_updated['V'] = V

        out = sampling_scheme(**params_updated)
        expected = np.append(frequencies, params_updated['scheme_param'])
        assert np.all(out == expected)
        assert (sum(out/sum(out)) - 1) < self.eps


    # test py behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5)
        ]
    )
    def test_py(self, V, frequencies, H):
        params_updated = self.params_py.copy()
        params_updated['frequencies'] = frequencies
        params_updated['H'] = H
        params_updated['V'] = V

        out = sampling_scheme(**params_updated)
        expected = np.append(
            frequencies - params_updated['sigma'],
            params_updated['scheme_param'] + params_updated['sigma'] * H
        )
        assert np.allclose(out, expected)
        assert (sum(out/sum(out)) - 1) < self.eps
    
    
    # test gn behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5)
        ]
    )
    def test_gn(self, V, frequencies, H):
        params_updated = self.params_gn.copy()
        params_updated['frequencies'] = frequencies
        params_updated['H'] = H
        params_updated['V'] = V
        out = sampling_scheme(**params_updated)
        
        expected = np.append(
            (params_updated['frequencies']+1) * (params_updated['V'] - params_updated['H'] + params_updated['gamma']),
            params_updated['H']* (params_updated['H'] - params_updated['gamma'])
        )
        assert np.all(out == expected)
        assert (sum(out/sum(out))-1) < self.eps
    
    
    # test py with sigma=0 is equivalent to dp
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5)
        ]
    )
    def test_py_equal_dp(self, V, frequencies, H):
        """Test that PY with sigma=0 is equivalent to DP"""
        params_py = self.params_py.copy()
        params_py['sigma']=0
        params_py['V'] = V
        params_py['frequencies'] = frequencies
        params_py['H'] = H
        
        params_dp = self.params_dp.copy()
        params_dp['V'] = V
        params_dp['frequencies'] = frequencies
        params_dp['H'] = H
        
        out_py = sampling_scheme(**params_py)
        out_dp = sampling_scheme(**params_dp)
        
        assert np.all(out_py == out_dp)
