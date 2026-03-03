import torch
import pytest


class TestStreamingWithCurriculum:
    def test_full_pipeline_narrowband(self):
        """End-to-end: benchmark config -> streaming loader -> curriculum."""
        from spectra.curriculum import CurriculumSchedule
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        from spectra.impairments import AWGN, Compose

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
                num_samples=16,
                num_iq_samples=256,
                sample_rate=1e6,
                impairments=Compose([AWGN(snr_range=params.get("snr_range", (10, 20)))]),
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            curriculum=schedule,
            batch_size=8,
        )

        # Epoch 0 and epoch 9 should produce different data
        data_e0, _ = next(iter(loader.epoch(0)))
        data_e9, _ = next(iter(loader.epoch(9)))
        assert data_e0.shape == (8, 2, 256)
        assert not torch.equal(data_e0, data_e9)

    def test_streaming_without_curriculum(self):
        """Streaming with no curriculum still varies per epoch."""
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
            num_epochs=100,
            batch_size=8,
        )

        # Collect data from 5 different epochs
        epoch_data = []
        for e in range(5):
            data, _ = next(iter(loader.epoch(e)))
            epoch_data.append(data)

        # All epochs should differ
        for i in range(4):
            assert not torch.equal(epoch_data[i], epoch_data[i + 1])

    def test_training_loop_pattern(self):
        """Verify the intended usage pattern works."""
        from spectra.streaming import StreamingDataLoader
        from spectra.curriculum import CurriculumSchedule
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK
        from spectra.impairments import AWGN, Compose

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )

        def factory(params):
            snr = params.get("snr_range", (10, 20))
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=128,
                sample_rate=1e6,
                impairments=Compose([AWGN(snr_range=snr)]),
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=3,
            curriculum=schedule,
            batch_size=4,
        )

        total_batches = 0
        for epoch in range(3):
            for batch_data, batch_labels in loader.epoch(epoch):
                total_batches += 1
                assert batch_data.ndim == 3

        assert total_batches == 6  # 3 epochs * 2 batches (8 samples / 4 batch_size)
