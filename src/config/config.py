"""
Configuration using Pydantic BaseModel for type-safe settings.
"""

import os
from pydantic import BaseModel

class Settings(BaseModel):
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    num_steps: int = 100_000
    epsilons: tuple[float, float, float] = (0.1, 0.5, 1.0)
    n_runs: int = 5


settings = Settings()

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'docs', '02-results')