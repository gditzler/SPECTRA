import torch


class TestCNNAMC:
    def test_forward_shape(self):
        from spectra.models import CNNAMC

        model = CNNAMC(num_classes=18)
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 18)

    def test_softmax_sums_to_one(self):
        from spectra.models import CNNAMC

        model = CNNAMC(num_classes=10)
        model.eval()
        x = torch.randn(2, 2, 512)
        with torch.no_grad():
            out = torch.softmax(model(x), dim=1)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_various_num_classes(self):
        from spectra.models import CNNAMC

        for nc in [2, 5, 18, 64]:
            model = CNNAMC(num_classes=nc)
            x = torch.randn(1, 2, 256)
            assert model(x).shape == (1, nc)

    def test_various_input_sizes(self):
        from spectra.models import CNNAMC

        model = CNNAMC(num_classes=8)
        for n in [128, 256, 512, 1024, 2048]:
            x = torch.randn(2, 2, n)
            assert model(x).shape == (2, 8)


class TestResNetAMC:
    def test_forward_shape(self):
        from spectra.models import ResNetAMC

        model = ResNetAMC(num_classes=18)
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, 18)

    def test_softmax_sums_to_one(self):
        from spectra.models import ResNetAMC

        model = ResNetAMC(num_classes=10)
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = torch.softmax(model(x), dim=1)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_various_num_classes(self):
        from spectra.models import ResNetAMC

        for nc in [2, 5, 18]:
            model = ResNetAMC(num_classes=nc)
            model.eval()
            x = torch.randn(1, 1, 32, 32)
            with torch.no_grad():
                assert model(x).shape == (1, nc)

    def test_multi_channel_input(self):
        from spectra.models import ResNetAMC

        model = ResNetAMC(num_classes=8, in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        assert model(x).shape == (2, 8)
