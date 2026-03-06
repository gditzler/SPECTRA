# spectra.impairments

Channel impairment transforms. All classes implement `Transform.__call__(iq, desc, **kwargs)`
and can be chained with `Compose`.

## Base & Composition

::: spectra.impairments.base.Transform

::: spectra.impairments.compose.Compose

## Noise

::: spectra.impairments.awgn.AWGN
::: spectra.impairments.colored_noise.ColoredNoise

## Frequency Effects

::: spectra.impairments.frequency_offset.FrequencyOffset
::: spectra.impairments.frequency_drift.FrequencyDrift
::: spectra.impairments.doppler.DopplerShift

## Phase Effects

::: spectra.impairments.phase_noise.PhaseNoise
::: spectra.impairments.phase_offset.PhaseOffset
::: spectra.impairments.spectral_inversion.SpectralInversion

## Hardware Impairments

::: spectra.impairments.iq_imbalance.IQImbalance
::: spectra.impairments.dc_offset.DCOffset
::: spectra.impairments.quantization.Quantization
::: spectra.impairments.sample_rate_offset.SampleRateOffset
::: spectra.impairments.passband_ripple.PassbandRipple

## Channel Models

::: spectra.impairments.fading.RayleighFading
::: spectra.impairments.fading.RicianFading

## MIMO Channels

::: spectra.impairments.mimo_channel.MIMOChannel

## MIMO Utilities

::: spectra.impairments.mimo_utils.steering_vector
::: spectra.impairments.mimo_utils.exponential_correlation
::: spectra.impairments.mimo_utils.kronecker_correlation

## 3GPP TDL Channel Models

::: spectra.impairments.tdl_channel.TDLChannel

## Power Amplifier Nonlinearity

::: spectra.impairments.power_amplifier.RappPA
::: spectra.impairments.power_amplifier.SalehPA

## Timing Impairments

::: spectra.impairments.timing.FractionalDelay
::: spectra.impairments.timing.SamplingJitter

## Interference

::: spectra.impairments.adjacent_channel.AdjacentChannelInterference
::: spectra.impairments.intermod.IntermodulationProducts
