"""Launch the team backend with the combined engine (hand-made + WavLM)."""
import ml.inference as _mi
from hm.combined import VoiceShieldInference as _Combined

_mi.VoiceShieldInference = _Combined  # backend.main imports this name

from backend.main import app  # noqa: E402
