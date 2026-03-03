from typing import Any, Dict, Optional, Tuple


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at parameter t."""
    return a + (b - a) * t


class CurriculumSchedule:
    """Maps training progress [0.0, 1.0] to parameter ranges via linear interpolation.

    Parameters
    ----------
    snr_range : dict, optional
        {"start": (lo, hi), "end": (lo, hi)} — SNR range in dB.
    num_signals : dict, optional
        {"start": (min, max), "end": (min, max)} — signal count range (wideband).
    impairments : dict, optional
        {"name": {"start": severity, "end": severity}} — per-impairment severity.
    """

    def __init__(
        self,
        snr_range: Optional[Dict[str, Tuple[float, float]]] = None,
        num_signals: Optional[Dict[str, Tuple[int, int]]] = None,
        impairments: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.snr_range = snr_range
        self.num_signals = num_signals
        self.impairments = impairments

    def at(self, progress: float) -> Dict[str, Any]:
        """Return interpolated parameters at the given progress.

        Parameters
        ----------
        progress : float
            Training progress in [0.0, 1.0]. Values outside this range are clamped.

        Returns
        -------
        dict
            Interpolated parameter values. Only includes fields that were configured.
        """
        t = max(0.0, min(1.0, progress))
        result: Dict[str, Any] = {}

        if self.snr_range is not None:
            s = self.snr_range["start"]
            e = self.snr_range["end"]
            result["snr_range"] = (_lerp(s[0], e[0], t), _lerp(s[1], e[1], t))

        if self.num_signals is not None:
            s = self.num_signals["start"]
            e = self.num_signals["end"]
            result["num_signals"] = (
                round(_lerp(s[0], e[0], t)),
                round(_lerp(s[1], e[1], t)),
            )

        if self.impairments is not None:
            imp_result = {}
            for name, cfg in self.impairments.items():
                imp_result[name] = _lerp(cfg["start"], cfg["end"], t)
            result["impairments"] = imp_result

        return result
