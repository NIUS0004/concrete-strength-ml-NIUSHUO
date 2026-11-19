"""Shared schema. Ingredient quantities are kg/m^3; age is days."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
            'coarse_aggregate', 'fine_aggregate', 'age']
TARGET = 'strength_mpa'
SEED = 42
