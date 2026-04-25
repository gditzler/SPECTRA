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

## Alignment & Domain Adaptation

::: spectra.transforms.alignment.DCRemove
::: spectra.transforms.alignment.Resample
::: spectra.transforms.alignment.PowerNormalize
::: spectra.transforms.alignment.AGCNormalize
::: spectra.transforms.alignment.ClipNormalize
::: spectra.transforms.alignment.BandpassAlign
::: spectra.transforms.alignment.NoiseFloorMatch
::: spectra.transforms.alignment.NoiseProfileTransfer
::: spectra.transforms.alignment.SpectralWhitening
::: spectra.transforms.alignment.ReceiverEQ

## Time-Frequency (Additional)

::: spectra.transforms.cwd.CWD
::: spectra.transforms.reassigned_gabor.ReassignedGabor
::: spectra.transforms.instantaneous_frequency.InstantaneousFrequency

## Other Representations

::: spectra.transforms.snapshot.ToSnapshotMatrix
