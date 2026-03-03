import pytest


class TestCurriculumSchedule:
    def test_snr_at_start(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        params = schedule.at(0.0)
        assert params["snr_range"] == pytest.approx((20.0, 30.0))

    def test_snr_at_end(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        params = schedule.at(1.0)
        assert params["snr_range"] == pytest.approx((0.0, 10.0))

    def test_snr_at_midpoint(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        params = schedule.at(0.5)
        assert params["snr_range"] == pytest.approx((10.0, 20.0))

    def test_num_signals_interpolation(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            num_signals={"start": (1, 2), "end": (4, 8)},
        )
        params = schedule.at(0.5)
        # Linear interp: (2.5, 5.0) -> rounded to ints: (2, 5)
        assert params["num_signals"] == (2, 5)

    def test_num_signals_at_boundaries(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            num_signals={"start": (1, 2), "end": (4, 8)},
        )
        assert schedule.at(0.0)["num_signals"] == (1, 2)
        assert schedule.at(1.0)["num_signals"] == (4, 8)

    def test_impairment_scheduling(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            impairments={
                "phase_offset": {"start": 0.0, "end": 0.5},
                "iq_imbalance": {"start": 0.0, "end": 0.2},
            },
        )
        params = schedule.at(0.5)
        assert params["impairments"]["phase_offset"] == pytest.approx(0.25)
        assert params["impairments"]["iq_imbalance"] == pytest.approx(0.1)

    def test_impairments_at_boundaries(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            impairments={"phase_offset": {"start": 0.0, "end": 1.0}},
        )
        assert schedule.at(0.0)["impairments"]["phase_offset"] == pytest.approx(0.0)
        assert schedule.at(1.0)["impairments"]["phase_offset"] == pytest.approx(1.0)

    def test_combined_schedule(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
            num_signals={"start": (1, 2), "end": (3, 6)},
            impairments={"phase_offset": {"start": 0.0, "end": 0.5}},
        )
        params = schedule.at(0.25)
        assert params["snr_range"] == pytest.approx((15.0, 25.0))
        assert params["num_signals"] == (2, 3)
        assert params["impairments"]["phase_offset"] == pytest.approx(0.125)

    def test_none_fields_omitted(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)})
        params = schedule.at(0.5)
        assert "snr_range" in params
        assert "num_signals" not in params
        assert "impairments" not in params

    def test_progress_clamped(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        assert schedule.at(-0.5)["snr_range"] == pytest.approx((20.0, 30.0))
        assert schedule.at(1.5)["snr_range"] == pytest.approx((0.0, 10.0))

    def test_empty_schedule(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule()
        params = schedule.at(0.5)
        assert params == {}
