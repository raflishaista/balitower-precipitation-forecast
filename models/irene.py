"""
irene — ConvGRU ensemble for radar-based lightning nowcasting.

Loads from HuggingFace Hub (it4lia/irene) and runs ensemble inference on past
radar rain-rate fields (mm/h) to produce future radar frames.

Install:
    pip install convgru-ensemble
"""
from convgru_ensemble import RadarLightningModel

# Load from HuggingFace Hub
model = RadarLightningModel.from_pretrained("it4lia/irene")

# Run inference on past radar data (rain rate in mm/h)
import numpy as np
past = np.random.rand(6, 256, 256).astype(np.float32)  # 6 past timesteps
forecasts = model.predict(past, forecast_steps=12, ensemble_size=10)
# forecasts.shape = (10, 12, 256, 256) — 10 ensemble members, 12 future steps