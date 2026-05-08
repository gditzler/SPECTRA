# Verification References

Canonical citations for every theoretical formula, tolerance, and spec
constraint used in `examples/verification/`.  Cite by key
(e.g. `proakis2008:eq4.3-15`) — the renderer expands the key into the
short form (`Proakis 2008, Eq. 4.3-15, p.193`) in result tables.

The parser in `_verify_helpers.py` reads this file at script startup.
**Unresolved citation keys raise an error.**

## [proakis2008]
- Authors: J. G. Proakis and M. Salehi
- Title:   *Digital Communications*, 5th edition
- Year:    2008
- Pub:     McGraw-Hill
- ISBN:    978-0072957167
- Loci used:
  - eq4.3-13, p.191  — BER coherent BPSK over AWGN: P_b = Q(sqrt(2·Eb/N0))
  - eq4.3-15, p.193  — SER coherent M-PSK over AWGN: P_s ≈ 2·Q(sqrt(2γ_s)·sin(π/M))
  - eq4.3-30, p.205  — SER square M-QAM over AWGN
  - eq4.4-43, p.227  — BER MSK / GMSK approximation over AWGN
  - eq9.2-37, p.560  — PSD of root-raised-cosine pulse
  - §4.4-3,   p.222  — CPM modulation index h=1/2 for MSK family
  - §9.2,     p.555  — PAPR of pulse-shaped linear modulations

## [sklar2001]
- Authors: B. Sklar
- Title:   *Digital Communications: Fundamentals and Applications*, 2nd ed.
- Year:    2001
- Pub:     Prentice Hall
- ISBN:    978-0130847881
- Loci used:
  - §3.5, eq3.74    — Occupied bandwidth: B = (1+α)·R_s for RRC pulse shaping
  - §3.5            — Gray-coded QPSK / QAM constellations

## [levanon2004]
- Authors: N. Levanon and E. Mozeson
- Title:   *Radar Signals*
- Year:    2004
- Pub:     Wiley-IEEE
- ISBN:    978-0471473787
- Loci used:
  - eq3.32          — Barker-N autocorrelation: peak/max-sidelobe = N
  - eq5.5           — LFM matched-filter compression gain = 10·log10(TBP)
  - §3              — Barker codes: detection at low SNR
  - §4.2            — LFM ambiguity function knife-edge property
  - §5              — Pulse compression: range resolution
  - Tab.6.1         — Canonical Barker code sequences

## [3gpp_38_211]
- Org:     3GPP
- Title:   TS 38.211 v17.4.0 — Physical channels and modulation
- Year:    2022
- URL:     https://www.3gpp.org/dynareport/38211.htm
- Loci used:
  - §7.4.2.2.1      — PSS sequence d_PSS(n) generated from m-sequence x(i)
  - §7.4.2.2.2      — PSS frequency-domain placement
  - §7.4.2.3.1      — SSS sequence d_SSS(n) from Gold-sequence pair
  - §7.4.2.3.2      — SSS frequency-domain placement

## [3gpp_38_104]
- Org:     3GPP
- Title:   TS 38.104 v17.7.0 — Base Station radio transmission and reception
- Year:    2022
- URL:     https://www.3gpp.org/dynareport/38104.htm
- Loci used:
  - T6.6.3.1-1      — ACLR limits Cat-A NR base station (≥45 dB)
  - §B.2            — EVM measurement procedure (RMS over equalized symbols)

## [rtca_do260b]
- Org:     RTCA
- Title:   DO-260B — MOPS for 1090 MHz Extended Squitter ADS-B
- Year:    2009
- Loci used:
  - §2.2.3.2.1.2    — CRC-24 generator polynomial G(x)=x²⁴+x²³+x¹⁸+...+1 (0x1FFF409)
  - §2.2.3.2.2      — PPM modulation, 1 µs preamble, 112 µs message
  - §2.2.4          — Spectrum mask

## [itu_sm_328]
- Org:     ITU-R
- Title:   Recommendation SM.328-11 — Spectra and bandwidth of emissions
- Loci used:
  - §3              — Occupied bandwidth definition (X% containment, X=99 standard)

## [laurent1986]
- Authors: P. Laurent
- Title:   "Exact and approximate construction of digital phase modulations
            by superposition of amplitude modulated pulses"
- Pub:     IEEE Transactions on Communications, vol. 34, no. 2, pp. 150–160
- Year:    1986
- DOI:     10.1109/TCOM.1986.1096498
- Loci used:
  - §III            — Laurent decomposition AMP main pulse for GMSK PSD

## [vandeBeek1997]
- Authors: J.-J. van de Beek, M. Sandell, P. O. Börjesson
- Title:   "ML Estimation of Time and Frequency Offset in OFDM Systems"
- Pub:     IEEE Transactions on Signal Processing, vol. 45, no. 7, pp. 1800–1805
- Year:    1997
- DOI:     10.1109/78.599949
- Loci used:
  - §III            — Cyclic-prefix correlation peak at lag = N_FFT

## [han2005]
- Authors: S. H. Han and J. H. Lee
- Title:   "An overview of peak-to-average power ratio reduction techniques
            for multicarrier transmission"
- Pub:     IEEE Wireless Communications, vol. 12, no. 2, pp. 56–65
- Year:    2005
- DOI:     10.1109/MWC.2005.1421929
- Loci used:
  - §I              — OFDM PAPR distribution (Gaussian approx, Rayleigh envelope)
