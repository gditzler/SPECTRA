import yaml
from spectra.cli.config_builder import (
    IMPAIRMENT_PRESETS,
    WAVEFORM_CATEGORIES,
    build_config,
    default_num_samples,
    default_seeds,
    get_impairment_registry,
    get_waveform_registry,
    serialize_config,
    validate_config,
)


class TestWaveformRegistry:
    def test_registry_not_empty(self):
        reg = get_waveform_registry()
        assert len(reg) > 60

    def test_registry_contains_bpsk(self):
        reg = get_waveform_registry()
        assert "BPSK" in reg

    def test_registry_matches_all(self):
        from spectra import waveforms as wmod
        from spectra.cli.config_builder import _WAVEFORM_EXCLUDES

        reg = get_waveform_registry()
        assert set(reg.keys()) == set(wmod.__all__) - _WAVEFORM_EXCLUDES


class TestImpairmentRegistry:
    def test_registry_not_empty(self):
        reg = get_impairment_registry()
        assert len(reg) > 15

    def test_excludes_compose(self):
        reg = get_impairment_registry()
        assert "Compose" not in reg

    def test_excludes_utility_functions(self):
        reg = get_impairment_registry()
        for name in ["exponential_correlation", "kronecker_correlation", "steering_vector"]:
            assert name not in reg

    def test_contains_awgn(self):
        reg = get_impairment_registry()
        assert "AWGN" in reg


class TestWaveformCategories:
    def test_categories_cover_all_waveforms(self):
        from spectra import waveforms as wmod
        from spectra.cli.config_builder import _WAVEFORM_EXCLUDES

        all_categorized = set()
        for names in WAVEFORM_CATEGORIES.values():
            all_categorized.update(names)
        assert all_categorized == set(wmod.__all__) - _WAVEFORM_EXCLUDES

    def test_no_duplicates_across_categories(self):
        seen = set()
        for names in WAVEFORM_CATEGORIES.values():
            for n in names:
                assert n not in seen, f"{n} in multiple categories"
                seen.add(n)

    def test_all_category_entries_exist_in_registry(self):
        reg = get_waveform_registry()
        for cat, names in WAVEFORM_CATEGORIES.items():
            for n in names:
                assert n in reg, f"{n} in '{cat}' not in registry"


class TestImpairmentPresets:
    def test_presets_not_empty(self):
        assert len(IMPAIRMENT_PRESETS) >= 3

    def test_clean_has_only_awgn(self):
        assert len(IMPAIRMENT_PRESETS["clean"]) == 1
        assert IMPAIRMENT_PRESETS["clean"][0]["type"] == "AWGN"

    def test_all_preset_types_in_registry(self):
        reg = get_impairment_registry()
        for preset, entries in IMPAIRMENT_PRESETS.items():
            for entry in entries:
                assert entry["type"] in reg, f"'{entry['type']}' in '{preset}' not in registry"


class TestBuildConfig:
    def test_narrowband_has_required_keys(self):
        cfg = build_config(
            name="test",
            task="narrowband",
            waveform_pool=[{"type": "BPSK"}, {"type": "QPSK"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1_000_000,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        for key in [
            "name",
            "version",
            "task",
            "sample_rate",
            "num_iq_samples",
            "num_samples",
            "seed",
            "waveform_pool",
            "snr_range",
            "impairments",
        ]:
            assert key in cfg

    def test_narrowband_has_no_scene(self):
        cfg = build_config(
            name="t",
            task="narrowband",
            waveform_pool=[{"type": "BPSK"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        assert "scene" not in cfg

    def test_wideband_has_scene(self):
        cfg = build_config(
            name="t",
            task="wideband",
            waveform_pool=[{"type": "BPSK"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
            scene={
                "capture_bandwidth": 500_000,
                "capture_duration": 0.001,
                "num_signals": [1, 3],
            },
        )
        assert "scene" in cfg
        assert cfg["scene"]["capture_bandwidth"] == 500_000


class TestValidateConfig:
    def test_valid_config_no_errors(self):
        cfg = build_config(
            name="t",
            task="narrowband",
            waveform_pool=[{"type": "BPSK"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        assert validate_config(cfg) == []

    def test_unknown_waveform_caught(self):
        cfg = build_config(
            name="t",
            task="narrowband",
            waveform_pool=[{"type": "FakeWaveform"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        errors = validate_config(cfg)
        assert any("FakeWaveform" in e for e in errors)

    def test_unknown_impairment_caught(self):
        cfg = build_config(
            name="t",
            task="narrowband",
            waveform_pool=[{"type": "BPSK"}],
            impairments=[{"type": "FakeImpairment"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        errors = validate_config(cfg)
        assert any("FakeImpairment" in e for e in errors)


class TestSerializeConfig:
    def test_roundtrip(self):
        cfg = build_config(
            name="test",
            task="narrowband",
            waveform_pool=[{"type": "BPSK"}, {"type": "QPSK"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        yaml_str = serialize_config(cfg)
        loaded = yaml.safe_load(yaml_str)
        assert loaded["name"] == "test"
        assert len(loaded["waveform_pool"]) == 2

    def test_params_preserved(self):
        cfg = build_config(
            name="t",
            task="narrowband",
            waveform_pool=[{"type": "FSK", "params": {"order": 4}}],
            impairments=[{"type": "AWGN"}],
            snr_range=(-10, 30),
            sample_rate=1e6,
            num_iq_samples=1024,
            num_samples=default_num_samples(),
            seed=default_seeds(),
        )
        loaded = yaml.safe_load(serialize_config(cfg))
        assert loaded["waveform_pool"][0]["params"]["order"] == 4


class TestDefaults:
    def test_default_seeds(self):
        s = default_seeds()
        assert set(s.keys()) == {"train", "val", "test"}

    def test_default_num_samples(self):
        n = default_num_samples()
        assert set(n.keys()) == {"train", "val", "test"}
        assert n["train"] > n["val"]


class TestIntegration:
    def test_generated_config_loads_with_load_benchmark(self, tmp_path):
        cfg = build_config(
            name="integration-test",
            task="narrowband",
            waveform_pool=[{"type": "BPSK"}, {"type": "QPSK"}],
            impairments=[
                {"type": "FrequencyOffset", "params": {"max_offset": 50_000}},
                {"type": "AWGN"},
            ],
            snr_range=(-10, 30),
            sample_rate=1_000_000,
            num_iq_samples=1024,
            num_samples={"train": 100, "val": 50, "test": 50},
            seed=default_seeds(),
        )
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(serialize_config(cfg))

        from spectra.benchmarks.loader import load_benchmark

        ds = load_benchmark(str(yaml_path), "train")
        assert len(ds) == 100
        sample = ds[0]
        assert sample[0].shape == (2, 1024)

    def test_wideband_config_loads(self, tmp_path):
        cfg = build_config(
            name="wb-test",
            task="wideband",
            waveform_pool=[{"type": "BPSK"}, {"type": "FM"}],
            impairments=[{"type": "AWGN"}],
            snr_range=(0, 20),
            sample_rate=1_000_000,
            num_iq_samples=2048,
            num_samples={"train": 50, "val": 20, "test": 20},
            seed=default_seeds(),
            scene={
                "capture_bandwidth": 500_000,
                "capture_duration": 0.002,
                "num_signals": [1, 3],
            },
        )
        yaml_path = tmp_path / "wb-test.yaml"
        yaml_path.write_text(serialize_config(cfg))

        from spectra.benchmarks.loader import load_benchmark

        ds = load_benchmark(str(yaml_path), "train")
        assert len(ds) == 50
