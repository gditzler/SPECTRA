import numpy as np
import torch
import pytest


class TestStreamingDataLoader:
    def test_epoch_returns_dataloader(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=32,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            batch_size=8,
        )
        dl = loader.epoch(0)
        assert isinstance(dl, torch.utils.data.DataLoader)

    def test_epoch_produces_batches(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=16,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=5,
            batch_size=8,
        )
        batches = list(loader.epoch(0))
        assert len(batches) == 2  # 16 samples / 8 batch_size
        data, labels = batches[0]
        assert data.shape == (8, 2, 256)

    def test_different_epochs_produce_different_data(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            batch_size=8,
        )
        data_e0, _ = next(iter(loader.epoch(0)))
        data_e1, _ = next(iter(loader.epoch(1)))
        assert not torch.equal(data_e0, data_e1)

    def test_same_epoch_is_deterministic(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            batch_size=8,
        )
        data_a, _ = next(iter(loader.epoch(3)))
        data_b, _ = next(iter(loader.epoch(3)))
        torch.testing.assert_close(data_a, data_b)

    def test_same_base_seed_reproduces_across_instances(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader1 = StreamingDataLoader(
            dataset_factory=factory, base_seed=42, num_epochs=10, batch_size=8
        )
        loader2 = StreamingDataLoader(
            dataset_factory=factory, base_seed=42, num_epochs=10, batch_size=8
        )
        d1, _ = next(iter(loader1.epoch(5)))
        d2, _ = next(iter(loader2.epoch(5)))
        torch.testing.assert_close(d1, d2)

    def test_different_base_seeds_differ(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader1 = StreamingDataLoader(
            dataset_factory=factory, base_seed=42, num_epochs=10, batch_size=8
        )
        loader2 = StreamingDataLoader(
            dataset_factory=factory, base_seed=99, num_epochs=10, batch_size=8
        )
        d1, _ = next(iter(loader1.epoch(0)))
        d2, _ = next(iter(loader2.epoch(0)))
        assert not torch.equal(d1, d2)


class TestStreamingDataLoaderWithCurriculum:
    def test_curriculum_params_passed_to_factory(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.curriculum import CurriculumSchedule

        captured_params = []

        def factory(params):
            captured_params.append(params.copy())
            from spectra.datasets import NarrowbandDataset
            from spectra.waveforms import QPSK

            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=4,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            curriculum=schedule,
            batch_size=4,
        )

        # Epoch 0 -> progress=0.0 -> snr_range=(20, 30)
        list(loader.epoch(0))
        assert captured_params[0]["snr_range"] == pytest.approx((20.0, 30.0))

        # Epoch 9 -> progress=1.0 -> snr_range=(0, 10)
        list(loader.epoch(9))
        assert captured_params[1]["snr_range"] == pytest.approx((0.0, 10.0))

    def test_no_curriculum_passes_seed_only(self):
        captured_params = []

        def factory(params):
            captured_params.append(params.copy())
            from spectra.datasets import NarrowbandDataset
            from spectra.waveforms import QPSK

            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=4,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        from spectra.streaming import StreamingDataLoader

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=5,
            batch_size=4,
        )
        list(loader.epoch(0))
        assert "seed" in captured_params[0]
        assert "snr_range" not in captured_params[0]

    def test_single_epoch_progress_is_zero(self):
        captured_params = []

        def factory(params):
            captured_params.append(params.copy())
            from spectra.datasets import NarrowbandDataset
            from spectra.waveforms import QPSK

            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=4,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        from spectra.streaming import StreamingDataLoader
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=1,
            curriculum=schedule,
            batch_size=4,
        )
        list(loader.epoch(0))
        # Single epoch: progress=0.0
        assert captured_params[0]["snr_range"] == pytest.approx((20.0, 30.0))
