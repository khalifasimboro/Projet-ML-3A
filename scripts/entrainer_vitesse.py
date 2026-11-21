"""Entraîne le modèle qui reconnaît la vitesse de rotation (modeles/vitesse.joblib).

    .venv/Scripts/python scripts/entrainer_vitesse.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ballguard import vitesse  # noqa: E402

if __name__ == "__main__":
    print(f"exactitude sur le jeu de test : {vitesse.entrainer():.2%}")
