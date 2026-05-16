"""Four codecs probed in the paper. Imports are lazy so users only pay
for the dependencies of the codec they actually use.
"""

from typing import Optional

from .base import BaseCodec


def load_encodec(bandwidth: float = 12.0, device: str = "cuda") -> BaseCodec:
    from .encodec import EnCodec
    return EnCodec(bandwidth=bandwidth, device=device)


def load_dac(model_type: str = "24khz", device: str = "cuda") -> BaseCodec:
    from .dac import DAC
    return DAC(model_type=model_type, device=device)


def load_mimi(device: str = "cuda", add_semantic_to_acoustic: bool = True) -> BaseCodec:
    from .mimi import MIMI
    return MIMI(device=device, add_semantic_to_acoustic=add_semantic_to_acoustic)


def load_mimo(checkpoint: str, device: str = "cuda") -> BaseCodec:
    from .mimo import MIMO
    return MIMO(checkpoint=checkpoint, device=device)


def load_codec(name: str, device: str = "cuda", checkpoint: Optional[str] = None) -> BaseCodec:
    """Dispatch by name. ``name`` is one of ``encodec``, ``dac``, ``mimi``, ``mimo``."""
    n = name.lower()
    if n.startswith("encodec"):
        return load_encodec(device=device)
    if n.startswith("dac"):
        return load_dac(device=device)
    if n == "mimi":
        return load_mimi(device=device)
    if n == "mimo":
        if checkpoint is None:
            raise ValueError("MIMO requires `checkpoint=`.")
        return load_mimo(checkpoint=checkpoint, device=device)
    raise ValueError(f"Unknown codec: {name!r}")


__all__ = ["BaseCodec", "load_encodec", "load_dac", "load_mimi", "load_mimo", "load_codec"]
