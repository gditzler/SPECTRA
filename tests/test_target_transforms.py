import pytest


class TestClassIndex:
    def test_mapping(self):
        from spectra.transforms import ClassIndex

        ci = ClassIndex(["BPSK", "QPSK", "8PSK"])
        assert ci("BPSK") == 0
        assert ci("QPSK") == 1
        assert ci("8PSK") == 2

    def test_unknown_raises(self):
        from spectra.transforms import ClassIndex

        ci = ClassIndex(["BPSK"])
        with pytest.raises(KeyError):
            ci("QPSK")


class TestFamilyName:
    def test_known_labels(self):
        from spectra.transforms import FamilyName

        fn = FamilyName()
        assert fn("BPSK") == "psk"
        assert fn("16QAM") == "qam"
        assert fn("FSK") == "fsk"
        assert fn("LFM") == "radar"
        assert fn("FM") == "fm"

    def test_unknown_label(self):
        from spectra.transforms import FamilyName

        assert FamilyName()("Unknown123") == "unknown"


class TestFamilyIndex:
    def test_returns_int(self):
        from spectra.transforms import FamilyIndex

        fi = FamilyIndex()
        assert isinstance(fi("BPSK"), int)
        assert fi("BPSK") >= 0


class TestYOLOLabel:
    def test_format(self):
        from spectra.transforms import YOLOLabel

        yolo = YOLOLabel(image_width=100, image_height=100)
        targets = [{"bbox": [10, 20, 30, 40], "category_id": 2}]
        result = yolo(targets)
        assert len(result) == 1
        assert len(result[0]) == 5
        # cx = (10 + 15) / 100 = 0.25
        assert result[0][1] == pytest.approx(0.25)


class TestBoxesNormalize:
    def test_normalized(self):
        from spectra.transforms import BoxesNormalize

        bn = BoxesNormalize(image_width=200, image_height=100)
        targets = [{"bbox": [20, 10, 40, 20], "label": "QPSK"}]
        result = bn(targets)
        assert result[0]["bbox"][0] == pytest.approx(0.1)
        assert result[0]["bbox"][1] == pytest.approx(0.1)
