from abc import ABC, abstractmethod
from typing import Any, Dict, List


FAMILY_MAP = {
    "BPSK": "psk",
    "QPSK": "psk",
    "8PSK": "psk",
    "16PSK": "psk",
    "32PSK": "psk",
    "64PSK": "psk",
    "16QAM": "qam",
    "32QAM": "qam",
    "64QAM": "qam",
    "128QAM": "qam",
    "256QAM": "qam",
    "512QAM": "qam",
    "1024QAM": "qam",
    "FSK": "fsk",
    "4FSK": "fsk",
    "8FSK": "fsk",
    "16FSK": "fsk",
    "MSK": "fsk",
    "4MSK": "fsk",
    "8MSK": "fsk",
    "GMSK": "fsk",
    "4GMSK": "fsk",
    "8GMSK": "fsk",
    "GFSK": "fsk",
    "4GFSK": "fsk",
    "8GFSK": "fsk",
    "16GFSK": "fsk",
    "OOK": "ask",
    "4ASK": "ask",
    "8ASK": "ask",
    "16ASK": "ask",
    "32ASK": "ask",
    "64ASK": "ask",
    "AM-DSB-SC": "am",
    "AM-DSB": "am",
    "AM-LSB": "am",
    "AM-USB": "am",
    "FM": "fm",
    "Tone": "tone",
    "LFM": "radar",
    "Frank": "radar",
    "P1": "radar",
    "P2": "radar",
    "P3": "radar",
    "P4": "radar",
    "Costas": "radar",
    "ChirpSS": "chirp",
    "OFDM": "ofdm",
    "OFDM-72": "ofdm",
    "OFDM-128": "ofdm",
    "OFDM-180": "ofdm",
    "OFDM-256": "ofdm",
    "OFDM-300": "ofdm",
    "OFDM-512": "ofdm",
    "OFDM-600": "ofdm",
    "OFDM-900": "ofdm",
    "OFDM-1200": "ofdm",
    "OFDM-2048": "ofdm",
}


class TargetTransform(ABC):
    @abstractmethod
    def __call__(self, target: Any) -> Any:
        ...


class ClassIndex(TargetTransform):
    """Map label string to integer index."""

    def __init__(self, class_list: List[str]):
        self._class_map = {label: idx for idx, label in enumerate(class_list)}

    def __call__(self, label: str) -> int:
        return self._class_map[label]


class FamilyName(TargetTransform):
    """Map waveform label to family name."""

    def __call__(self, label: str) -> str:
        return FAMILY_MAP.get(label, "unknown")


class FamilyIndex(TargetTransform):
    """Map waveform label to family integer index."""

    def __init__(self):
        families = sorted(set(FAMILY_MAP.values()))
        self._family_map = {name: idx for idx, name in enumerate(families)}

    def __call__(self, label: str) -> int:
        family = FAMILY_MAP.get(label, "unknown")
        return self._family_map.get(family, -1)


class YOLOLabel(TargetTransform):
    """Convert COCO targets to YOLO format [class_id, cx, cy, w, h]."""

    def __init__(self, image_width: int, image_height: int):
        self._w = image_width
        self._h = image_height

    def __call__(self, targets: List[Dict]) -> List[List[float]]:
        yolo_labels = []
        for t in targets:
            bbox = t["bbox"]  # [x, y, w, h] in pixels
            x, y, w, h = bbox
            cx = (x + w / 2.0) / self._w
            cy = (y + h / 2.0) / self._h
            nw = w / self._w
            nh = h / self._h
            yolo_labels.append([t.get("category_id", 0), cx, cy, nw, nh])
        return yolo_labels


class BoxesNormalize(TargetTransform):
    """Normalize bounding boxes to [0, 1] range."""

    def __init__(self, image_width: int, image_height: int):
        self._w = image_width
        self._h = image_height

    def __call__(self, targets: List[Dict]) -> List[Dict]:
        normalized = []
        for t in targets:
            bbox = t["bbox"]
            new_bbox = [
                bbox[0] / self._w,
                bbox[1] / self._h,
                bbox[2] / self._w,
                bbox[3] / self._h,
            ]
            nt = dict(t)
            nt["bbox"] = new_bbox
            normalized.append(nt)
        return normalized
