import numpy as np
import pytest
import torch


class TestFileWriterABC:
    def test_cannot_instantiate(self):
        from spectra.utils.file_handlers.base_writer import FileWriter

        with pytest.raises(TypeError):
            FileWriter()


class TestTensorToComplex64:
    def test_2d_tensor(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        iq = _tensor_to_complex64(t)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)
        np.testing.assert_allclose(iq.real, [1.0, 2.0])
        np.testing.assert_allclose(iq.imag, [3.0, 4.0])

    def test_1d_complex_passthrough(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        result = _tensor_to_complex64(arr)
        np.testing.assert_array_equal(result, arr)

    def test_2d_ndarray(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        iq = _tensor_to_complex64(arr)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)
