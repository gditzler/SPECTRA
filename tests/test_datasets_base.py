from spectra.datasets._base import BaseIQDataset


class ConcreteDataset(BaseIQDataset):
    """Minimal concrete subclass for testing."""
    def __getitem__(self, index):
        rng = self._make_rng(index)
        return rng.random()


class TestBaseIQDataset:
    def test_len(self):
        ds = ConcreteDataset(num_samples=100, seed=42)
        assert len(ds) == 100

    def test_seed_none_defaults_to_zero(self):
        ds = ConcreteDataset(num_samples=10, seed=None)
        assert ds.seed == 0

    def test_deterministic_rng(self):
        ds = ConcreteDataset(num_samples=10, seed=123)
        assert ds[5] == ds[5]

    def test_different_indices_differ(self):
        ds = ConcreteDataset(num_samples=10, seed=123)
        assert ds[0] != ds[1]

    def test_different_seeds_differ(self):
        ds1 = ConcreteDataset(num_samples=10, seed=1)
        ds2 = ConcreteDataset(num_samples=10, seed=2)
        assert ds1[0] != ds2[0]
