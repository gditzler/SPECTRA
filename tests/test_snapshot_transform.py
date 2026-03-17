import numpy as np


def test_snapshot_matrix_shape():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    t = ToSnapshotMatrix()
    x = np.random.randn(4, 2, 128).astype(np.float32)  # [n_elem, 2, T]
    result = t(x)
    assert result.shape == (4, 128)


def test_snapshot_matrix_complex():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    t = ToSnapshotMatrix()
    x = np.random.randn(4, 2, 128).astype(np.float32)
    result = t(x)
    assert np.issubdtype(result.dtype, np.complexfloating)


def test_snapshot_matrix_values():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    t = ToSnapshotMatrix()
    i_channel = np.ones((3, 64), dtype=np.float32) * 2.0
    q_channel = np.ones((3, 64), dtype=np.float32) * 3.0
    x = np.stack([i_channel, q_channel], axis=1)  # (3, 2, 64)
    result = t(x)
    np.testing.assert_allclose(result.real, 2.0)
    np.testing.assert_allclose(result.imag, 3.0)
