"""Tests for target trajectory motion models."""
import numpy as np
import pytest


def test_cv_initial_state():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=1000.0, velocity=50.0, dt=1.0)
    state = cv.state_at(0)
    assert state.shape == (2,)
    assert state[0] == pytest.approx(1000.0)
    assert state[1] == pytest.approx(50.0)


def test_cv_propagation():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=1000.0, velocity=50.0, dt=1.0)
    state = cv.state_at(10)
    assert state[0] == pytest.approx(1500.0)
    assert state[1] == pytest.approx(50.0)


def test_cv_range_at():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=500.0, velocity=-20.0, dt=0.5)
    r = cv.range_at(4)
    assert r == pytest.approx(500.0 + (-20.0) * 4 * 0.5)


def test_cv_states_shape():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=100.0, velocity=10.0, dt=0.1)
    states = cv.states(20)
    assert states.shape == (20, 2)
    assert states[0, 0] == pytest.approx(100.0)
    assert states[0, 1] == pytest.approx(10.0)
    assert states[19, 0] == pytest.approx(100.0 + 10.0 * 19 * 0.1)


def test_ct_initial_state():
    from spectra.targets.trajectory import ConstantTurnRate
    ct = ConstantTurnRate(initial_range=2000.0, velocity=100.0, turn_rate=0.1, dt=1.0)
    state = ct.state_at(0)
    assert state.shape == (2,)
    assert state[0] == pytest.approx(2000.0)


def test_ct_range_changes():
    """CT range should vary non-linearly (sinusoidal character)."""
    from spectra.targets.trajectory import ConstantTurnRate
    ct = ConstantTurnRate(initial_range=5000.0, velocity=200.0, turn_rate=0.05, dt=1.0)
    ranges = [ct.range_at(t) for t in range(50)]
    assert max(ranges) != min(ranges)
    states = ct.states(50)
    rates = states[:, 1]
    assert not np.allclose(rates, rates[0])


def test_ct_states_shape():
    from spectra.targets.trajectory import ConstantTurnRate
    ct = ConstantTurnRate(initial_range=1000.0, velocity=50.0, turn_rate=0.1, dt=0.5)
    states = ct.states(30)
    assert states.shape == (30, 2)


def test_trajectory_protocol():
    from spectra.targets.trajectory import ConstantTurnRate, ConstantVelocity, Trajectory
    cv = ConstantVelocity(initial_range=100.0, velocity=10.0, dt=1.0)
    ct = ConstantTurnRate(initial_range=100.0, velocity=10.0, turn_rate=0.1, dt=1.0)
    assert isinstance(cv, Trajectory)
    assert isinstance(ct, Trajectory)
