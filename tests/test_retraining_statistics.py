import numpy as np

from src.retraining import _paired_p_value


def test_paired_significance_is_deterministic_and_bounded():
    champion=np.array([3.0,3.2,2.8,3.1,3.3]*8)
    challenger=np.array([2.0,2.2,1.8,2.1,2.3]*8)
    p1=_paired_p_value(champion,challenger,iterations=300)
    p2=_paired_p_value(champion,challenger,iterations=300)
    assert p1==p2
    assert 0<=p1<=1


def test_insufficient_promotion_samples_returns_none():
    assert _paired_p_value([1,2,3],[1,2,3],iterations=100) is None
