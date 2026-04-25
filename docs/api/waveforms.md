# spectra.waveforms

Waveform generators for 60+ signal types. All classes extend `Waveform` and are
safe to use across DataLoader workers — pass a `seed` to `generate()` for determinism.

## Base Classes

::: spectra.waveforms.base.Waveform

::: spectra.waveforms.rrc_base._RRCWaveformBase
    options:
      show_source: false

## PSK Family

::: spectra.waveforms.psk.BPSK
    options:
      show_source: false
::: spectra.waveforms.psk.QPSK
    options:
      show_source: false
::: spectra.waveforms.psk.PSK8
    options:
      show_source: false
::: spectra.waveforms.psk.PSK16
    options:
      show_source: false
::: spectra.waveforms.psk.PSK32
    options:
      show_source: false
::: spectra.waveforms.psk.PSK64
    options:
      show_source: false

## QAM Family

::: spectra.waveforms.qam._QAMBase
    options:
      show_source: false

All QAM variants (`QAM16`, `QAM32`, `QAM64`, `QAM128`, `QAM256`, `QAM512`, `QAM1024`)
inherit from `_QAMBase` and differ only in their constellation order.

## ASK / OOK Family

::: spectra.waveforms.ask.OOK
    options:
      show_source: false
::: spectra.waveforms.ask._ASKBase
    options:
      show_source: false

Variants: `ASK4`, `ASK8`, `ASK16`, `ASK32`, `ASK64`.

## FSK Family

::: spectra.waveforms.fsk.FSK
    options:
      show_source: false
::: spectra.waveforms.fsk.GFSK
    options:
      show_source: false
::: spectra.waveforms.fsk.GMSK
    options:
      show_source: false
::: spectra.waveforms.fsk.MSK
    options:
      show_source: false

Multi-level variants: `FSK4`, `FSK8`, `FSK16`, `GFSK4`, `GFSK8`, `GFSK16`,
`GMSK4`, `GMSK8`, `MSK4`, `MSK8`.

## AM / FM Family

::: spectra.waveforms.am.AMDSB
    options:
      show_source: false
::: spectra.waveforms.am.AMDSB_SC
    options:
      show_source: false
::: spectra.waveforms.am.AMLSB
    options:
      show_source: false
::: spectra.waveforms.am.AMUSB
    options:
      show_source: false
::: spectra.waveforms.fm.FM
    options:
      show_source: false

## OFDM Family

::: spectra.waveforms.ofdm.OFDM

Preconfigured variants with fixed subcarrier counts:
`OFDM72`, `OFDM128`, `OFDM180`, `OFDM300`, `OFDM512`, `OFDM600`, `OFDM1024`, `OFDM2048`.

## Radar / Spread-Spectrum

::: spectra.waveforms.lfm.LFM
    options:
      show_source: false
::: spectra.waveforms.chirpss.ChirpSS
    options:
      show_source: false
::: spectra.waveforms.dsss.DSSS_BPSK
    options:
      show_source: false
::: spectra.waveforms.zadoff_chu.ZadoffChu
    options:
      show_source: false
::: spectra.waveforms.barker.BarkerCode
    options:
      show_source: false
::: spectra.waveforms.polyphase.FrankCode
    options:
      show_source: false
::: spectra.waveforms.costas.CostasCode
    options:
      show_source: false
::: spectra.waveforms.polyphase.P1Code
    options:
      show_source: false
::: spectra.waveforms.polyphase.P2Code
    options:
      show_source: false
::: spectra.waveforms.polyphase.P3Code
    options:
      show_source: false
::: spectra.waveforms.polyphase.P4Code
    options:
      show_source: false

## Utility Waveforms

::: spectra.waveforms.tone.Tone
    options:
      show_source: false
::: spectra.waveforms.noise.Noise
    options:
      show_source: false

## 5G NR

::: spectra.waveforms.nr.NR_OFDM
    options:
      show_source: false
::: spectra.waveforms.nr.NR_SSB
    options:
      show_source: false
::: spectra.waveforms.nr.NR_PDSCH
    options:
      show_source: false
::: spectra.waveforms.nr.NR_PUSCH
    options:
      show_source: false
::: spectra.waveforms.nr.NR_PRACH
    options:
      show_source: false

## Aviation & Maritime Protocols

::: spectra.waveforms.aviation_maritime.ADSB
    options:
      show_source: false
::: spectra.waveforms.aviation_maritime.ModeS
    options:
      show_source: false
::: spectra.waveforms.aviation_maritime.AIS
    options:
      show_source: false
::: spectra.waveforms.aviation_maritime.ACARS
    options:
      show_source: false
::: spectra.waveforms.aviation_maritime.DME
    options:
      show_source: false
::: spectra.waveforms.aviation_maritime.ILS_Localizer
    options:
      show_source: false

## Radar (Additional)

::: spectra.waveforms.radar.PulsedRadar
    options:
      show_source: false
::: spectra.waveforms.radar.PulseDoppler
    options:
      show_source: false
::: spectra.waveforms.radar.FMCW
    options:
      show_source: false
::: spectra.waveforms.radar.SteppedFrequency
    options:
      show_source: false
::: spectra.waveforms.radar.NonlinearFM
    options:
      show_source: false
::: spectra.waveforms.radar.BarkerCodedPulse
    options:
      show_source: false
::: spectra.waveforms.radar.PolyphaseCodedPulse
    options:
      show_source: false

## Spread Spectrum (Additional)

::: spectra.waveforms.spread_spectrum.DSSS_QPSK
    options:
      show_source: false
::: spectra.waveforms.spread_spectrum.FHSS
    options:
      show_source: false
::: spectra.waveforms.spread_spectrum.THSS
    options:
      show_source: false
::: spectra.waveforms.spread_spectrum.CDMA_Forward
    options:
      show_source: false
::: spectra.waveforms.spread_spectrum.CDMA_Reverse
    options:
      show_source: false

## Multi-Function Emitters

::: spectra.waveforms.multifunction.scheduled_waveform.ScheduledWaveform
    options:
      show_source: false
::: spectra.waveforms.multifunction.schedule.StaticSchedule
    options:
      show_source: false
::: spectra.waveforms.multifunction.schedule.StochasticSchedule
    options:
      show_source: false
::: spectra.waveforms.multifunction.schedule.CognitiveSchedule
    options:
      show_source: false
::: spectra.waveforms.multifunction.schedule.SegmentSpec
    options:
      show_source: false
::: spectra.waveforms.multifunction.schedule.ModeSpec
    options:
      show_source: false
