# spectra.transforms

Spectral, CSP, and augmentation transforms compatible with PyTorch DataLoaders.

## Spectral Transforms

::: spectra.transforms.stft.STFT
::: spectra.transforms.spectrogram.Spectrogram
::: spectra.transforms.psd.PSD

## Cyclostationary Signal Processing

::: spectra.transforms.scd.SCD
::: spectra.transforms.scf.SCF
::: spectra.transforms.caf.CAF
::: spectra.transforms.cumulants.Cumulants
::: spectra.transforms.energy.EnergyDetector

## Normalization

::: spectra.transforms.normalize.Normalize

## Representation Conversion

::: spectra.transforms.complex_to_2d.ComplexTo2D

## Time-Frequency Representations

::: spectra.transforms.wvd.WVD
::: spectra.transforms.ambiguity.AmbiguityFunction

## Data Augmentations

::: spectra.transforms.augmentations.CutOut
::: spectra.transforms.augmentations.TimeReversal
::: spectra.transforms.augmentations.PatchShuffle
::: spectra.transforms.augmentations.MixUp
::: spectra.transforms.augmentations.CutMix

## Target Transforms

::: spectra.transforms.target_transforms.ClassIndex
::: spectra.transforms.target_transforms.FamilyIndex
::: spectra.transforms.target_transforms.FamilyName
::: spectra.transforms.target_transforms.BoxesNormalize
::: spectra.transforms.target_transforms.YOLOLabel
