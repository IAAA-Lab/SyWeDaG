import pytest

from utils.geospatial import calculate_distance_km


def test_same_point_is_zero_distance():
    assert calculate_distance_km(40.0, -3.0, 40.0, -3.0) == pytest.approx(0.0, abs=1e-6)

def test_known_distance_madrid_barcelona():
    # Madrid ~ (40.4168, -3.7038), Barcelona ~ (41.3874, 2.1686)
    # Great-circle distance is roughly 500 km.
    distance = calculate_distance_km(40.4168, -3.7038, 41.3874, 2.1686)
    assert 480 <= distance <= 520

def test_distance_is_symmetric():
    d1 = calculate_distance_km(40.0, -3.0, 41.0, -4.0)
    d2 = calculate_distance_km(41.0, -4.0, 40.0, -3.0)
    assert d1 == pytest.approx(d2)

def test_one_degree_latitude_is_about_111_km():
    distance = calculate_distance_km(0.0, 0.0, 1.0, 0.0)
    assert 110 <= distance <= 112
