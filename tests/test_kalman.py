"""Tests for generic Kalman filter and CV factory."""
import numpy as np
import pytest


def test_kf_initial_state():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    kf = KalmanFilter(F, H, Q, R)
    assert kf.state.shape == (2,)
    assert np.allclose(kf.state, 0.0)
    assert kf.covariance.shape == (2, 2)


def test_kf_custom_initial():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    x0 = np.array([100.0, 5.0])
    P0 = np.eye(2) * 10.0
    kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
    assert np.allclose(kf.state, x0)
    assert np.allclose(kf.covariance, P0)


def test_kf_predict():
    from spectra.tracking.kalman import KalmanFilter
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    x0 = np.array([100.0, 10.0])
    kf = KalmanFilter(F, H, Q, R, x0=x0)
    predicted = kf.predict()
    assert predicted[0] == pytest.approx(110.0)
    assert predicted[1] == pytest.approx(10.0)


def test_kf_update_moves_toward_measurement():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    x0 = np.array([100.0, 0.0])
    P0 = np.eye(2) * 100.0
    kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
    kf.predict()
    updated = kf.update(np.array([105.0]))
    assert updated[0] > 100.0
    assert updated[0] < 106.0


def test_kf_step():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    kf = KalmanFilter(F, H, Q, R, x0=np.array([0.0, 0.0]))
    state = kf.step(np.array([10.0]))
    assert state[0] > 0.0


def test_kf_run_batch():
    from spectra.tracking.kalman import KalmanFilter
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.1
    R = np.array([[1.0]])
    x0 = np.array([0.0, 10.0])
    kf = KalmanFilter(F, H, Q, R, x0=x0)
    measurements = np.array([[10.0 * t + np.random.default_rng(t).normal(0, 1)]
                              for t in range(1, 21)])
    states = kf.run(measurements)
    assert states.shape == (20, 2)
    assert abs(states[-1, 0] - 200.0) < 20.0


def test_cv_kf_factory():
    from spectra.tracking.kalman import ConstantVelocityKF, KalmanFilter
    kf = ConstantVelocityKF(dt=0.5, process_noise_std=1.0, measurement_noise_std=5.0)
    assert isinstance(kf, KalmanFilter)
    assert kf.state.shape == (2,)
    kf2 = ConstantVelocityKF(dt=0.5, process_noise_std=1.0, measurement_noise_std=5.0,
                               x0=np.array([100.0, 20.0]))
    pred = kf2.predict()
    assert pred[0] == pytest.approx(110.0)
    assert pred[1] == pytest.approx(20.0)


def test_cv_kf_tracks_linear_target():
    from spectra.tracking.kalman import ConstantVelocityKF
    kf = ConstantVelocityKF(dt=1.0, process_noise_std=0.5, measurement_noise_std=2.0)
    rng = np.random.default_rng(42)
    measurements = np.array([[100.0 + 20.0 * t + rng.normal(0, 2)]
                              for t in range(1, 51)])
    states = kf.run(measurements)
    assert abs(states[-1, 0] - 1100.0) < 30.0
    assert abs(states[-1, 1] - 20.0) < 5.0
