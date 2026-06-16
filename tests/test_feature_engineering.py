import numpy as np
from dmri_mci_ad.feature_engineering import annualized_relative_change


def test_annualized_relative_change_basic():
    out = annualized_relative_change([100], [90], [12])
    assert np.allclose(out, [-0.1])
