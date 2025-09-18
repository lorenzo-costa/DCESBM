import pytest
import sys

sys.path.append("src/analysis")
from models.baseline import Baseline


class TestInitMethod:
    """Test cases for the __init__ method parameter validation"""
    
    def setup_method(self):
        """Setup common test parameters"""
        self.valid_params = {
            'num_items': 100,
            'num_users': 50,
            'prior_a': 1,
            'prior_b': 1,
            'epsilon': 1e-6,
            'seed': 42,
            'verbose_users': False,
            'verbose_items': False,
            'device': 'cpu'
        }
    
    # User Clustering Tests
    def test_user_clustering_invalid_type(self):
        """Test invalid user_clustering type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'user_clustering': 1 
        })
        with pytest.raises(Exception, match='user clustering must be a list or array'):
            Baseline(**params)
    
    def test_user_clustering_length_mismatch(self):
        """Test user_clustering length mismatch"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'user_clustering': [0, 1, 0]  # Length != num_users (50)
        })
        with pytest.raises(Exception, match='user clustering length does not match number of users'):
            Baseline(**params)
    
    def test_user_clustering_negative_values(self):
        """Test user_clustering with negative values"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'user_clustering': [0, -1, 2] + [0]*47  
        })
        with pytest.raises(Exception, match='user clustering must be non-negative integers'):
            Baseline(**params)
    
    def test_user_clustering_empty_list(self):
        """Test user_clustering with empty list"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'user_clustering': [] 
        })
        with pytest.raises(Exception, match='user clustering length does not match number of users'):
            Baseline(**params)
     
    # Item Clustering Tests       
    def item_clustering_invalid_type(self):
        """Test invalid item_clustering type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'item_clustering': 1 
        })
        with pytest.raises(Exception, match='item clustering must be a list or array'):
            Baseline(**params)
    
    def test_item_clustering_length_mismatch(self):
        """Test item_clustering length mismatch"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'item_clustering': [0, 1, 0]  # Length != num_items (100)
        })
        with pytest.raises(Exception, match='item clustering length does not match number of items'):
            Baseline(**params)
    
    def test_item_clustering_negative_values(self):
        """Test item_clustering with negative values"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'item_clustering': [0, -1, 2] + [0]*97  
        })
        with pytest.raises(Exception, match='item clustering must be non-negative integers'):
            Baseline(**params)
    
    def test_item_clustering_empty_list(self):
        """Test item_clustering with empty list"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0,
            'item_clustering': [] 
        })
        with pytest.raises(Exception, match='item clustering length does not match number of items'):
            Baseline(**params)
    
    # DM Scheme Tests
    def test_dm_valid_parameters(self):
        """Test DM scheme with valid parameters"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 20,
            'sigma': -0.5
        })
        obj = Baseline(**params)  # Should not raise
        
    def test_dm_invalid_bar_h_users_type(self):
        """Test DM scheme with invalid bar_h_users type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10.5,  # Should be int
            'bar_h_items': 20,
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters users for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_users_negative(self):
        """Test DM scheme with negative bar_h_users"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': -5, # should be positive
            'bar_h_items': 20,
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters users for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_users_zero(self):
        """Test DM scheme with bar_h_users = 0"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 0,  # should be positive
            'bar_h_items': 20,
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters users for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_users_too_large(self):
        """Test DM scheme with bar_h_users > num_users"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 100,  # > num_users (50)
            'bar_h_items': 20,
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters users for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_items_type(self):
        """Test DM scheme with invalid bar_h_items type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 20.5,  # Should be int
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters items for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_items_negative(self):
        """Test DM scheme with negative bar_h_items"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': -10,
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters items for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_items_zero(self):
        """Test DM scheme with bar_h_items = 0"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 0,  # should be positive
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters items for DM'):
            Baseline(**params)
    
    def test_dm_invalid_bar_h_items_too_large(self):
        """Test DM scheme with bar_h_items > num_items"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 150,  # > num_items (100)
            'sigma': -0.5
        })
        with pytest.raises(Exception, match='provide valid maximum number of clusters items for DM'):
            Baseline(**params)
    
    def test_dm_invalid_sigma_type(self):
        """Test DM scheme with invalid sigma type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 20,
            'sigma': 'invalid'
        })
        with pytest.raises(Exception, match='provide valid sigma \\(-item_clustering\\) parameter for DM'):
            Baseline(**params)
    
    def test_dm_invalid_sigma_positive(self):
        """Test DM scheme with positive sigma (should be negative)"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 20,
            'sigma': 0.5  # Should be < 0
        })
        with pytest.raises(Exception, match='provide valid sigma \\(-item_clustering\\) parameter for DM'):
            Baseline(**params)
    
    def test_dm_invalid_sigma_zero(self):
        """Test DM scheme with sigma = 0 (should be < 0)"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DM',
            'bar_h_users': 10,
            'bar_h_items': 20,
            'sigma': 0
        })
        with pytest.raises(Exception, match='provide valid sigma \\(-item_clustering\\) parameter for DM'):
            Baseline(**params)
    
    # DP Scheme Tests
    def test_dp_valid_parameters(self):
        """Test DP scheme with valid parameters"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DP',
            'scheme_param': 2.5
        })
        obj = Baseline(**params)  # Should not raise
    
    def test_dp_invalid_scheme_param_type(self):
        """Test DP scheme with invalid scheme_param type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DP',
            'scheme_param': 'invalid'
        })
        with pytest.raises(Exception, match='provide valid concentration parameter for DP'):
            Baseline(**params)
    
    def test_dp_invalid_scheme_param_zero(self):
        """Test DP scheme with scheme_param = 0"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DP',
            'scheme_param': 0
        })
        with pytest.raises(Exception, match='provide valid concentration parameter for DP'):
            Baseline(**params)
    
    def test_dp_invalid_scheme_param_negative(self):
        """Test DP scheme with negative scheme_param"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'DP',
            'scheme_param': -1.5
        })
        with pytest.raises(Exception, match='provide valid concentration parameter for DP'):
            Baseline(**params)
    
    # PY Scheme Tests
    def test_py_valid_parameters(self):
        """Test PY scheme with valid parameters"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 1.0
        })
        obj = Baseline(**params)  # Should not raise
    
    def test_py_valid_parameters_sigma_zero(self, capsys):
        """Test PY scheme with sigma = 0 (prints warning)"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0,
            'scheme_param': 1.0
        })
        obj = Baseline(**params)  # Should not raise
        captured = capsys.readouterr()
        assert "note: for sigma=0 the PY reduces to DP" in captured.out
    
    def test_py_invalid_sigma_type(self):
        """Test PY scheme with invalid sigma type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 'invalid',
            'scheme_param': 1.0
        })
        with pytest.raises(Exception, match='provide sigma in \\[0, 1\\) for PY'):
            Baseline(**params)
    
    def test_py_invalid_sigma_negative(self):
        """Test PY scheme with negative sigma"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': -0.1,
            'scheme_param': 1.0
        })
        with pytest.raises(Exception, match='provide sigma in \\[0, 1\\) for PY'):
            Baseline(**params)
    
    def test_py_invalid_sigma_equal_one(self):
        """Test PY scheme with sigma = 1"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 1.0,
            'scheme_param': 1.0
        })
        with pytest.raises(Exception, match='provide sigma in \\[0, 1\\) for PY'):
            Baseline(**params)
    
    def test_py_invalid_sigma_greater_than_one(self):
        """Test PY scheme with sigma > 1"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 1.5,
            'scheme_param': 1.0
        })
        with pytest.raises(Exception, match='provide sigma in \\[0, 1\\) for PY'):
            Baseline(**params)
    
    def test_py_invalid_scheme_param_type(self):
        """Test PY scheme with invalid scheme_param type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': 'invalid'
        })
        with pytest.raises(Exception, match='provide valid user_clustering parameter for PY'):
            Baseline(**params)
    
    def test_py_invalid_scheme_param_too_small(self):
        """Test PY scheme with scheme_param <= -sigma"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': -0.5  # Equal to -sigma
        })
        with pytest.raises(Exception, match='provide valid user_clustering parameter for PY'):
            Baseline(**params)
    
    def test_py_invalid_scheme_param_less_than_negative_sigma(self):
        """Test PY scheme with scheme_param < -sigma"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.5,
            'scheme_param': -0.6  # Less than -sigma
        })
        with pytest.raises(Exception, match='provide valid user_clustering parameter for PY'):
            Baseline(**params)
    
    # GN Scheme Tests
    def test_gn_valid_parameters(self):
        """Test GN scheme with valid parameters"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 0.5
        })
        obj = Baseline(**params)  # Should not raise
    
    def test_gn_invalid_gamma_type(self):
        """Test GN scheme with invalid gamma type"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 'invalid'
        })
        with pytest.raises(Exception, match='please provide valid gamma paramter for GN'):
            Baseline(**params)
    
    def test_gn_invalid_gamma_int_type(self):
        """Test GN scheme with int gamma (should be float)"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 1  # Should be float
        })
        with pytest.raises(Exception, match='please provide valid gamma paramter for GN'):
            Baseline(**params)
    
    def test_gn_invalid_gamma_zero(self):
        """Test GN scheme with gamma = 0"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 0.0
        })
        with pytest.raises(Exception, match='please provide valid gamma paramter for GN'):
            Baseline(**params)
    
    def test_gn_invalid_gamma_negative(self):
        """Test GN scheme with negative gamma"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': -0.1
        })
        with pytest.raises(Exception, match='please provide valid gamma paramter for GN'):
            Baseline(**params)
    
    def test_gn_invalid_gamma_equal_one(self):
        """Test GN scheme with gamma = 1"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 1.0
        })
        with pytest.raises(Exception, match='please provide valid gamma paramter for GN'):
            Baseline(**params)
    
    def test_gn_invalid_gamma_greater_than_one(self):
        """Test GN scheme with gamma > 1"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 1.5
        })
        with pytest.raises(Exception, match='please provide valid gamma paramter for GN'):
            Baseline(**params)
    
    def test_py_boundary_sigma_almost_one(self):
        """Test PY scheme with sigma very close to 1"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'PY',
            'sigma': 0.9999999,
            'scheme_param': 1.0
        })
        obj = Baseline(**params)  # Should not raise
    
    def test_gn_boundary_gamma_almost_one(self):
        """Test GN scheme with gamma very close to 1"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 0.9999999
        })
        obj = Baseline(**params)  # Should not raise
    
    def test_gn_boundary_gamma_very_small(self):
        """Test GN scheme with very small positive gamma"""
        params = self.valid_params.copy()
        params.update({
            'scheme_type': 'GN',
            'gamma': 1e-10
        })
        obj = Baseline(**params)  # Should not raise