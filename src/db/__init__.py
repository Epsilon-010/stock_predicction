"""Database ORM layer — SQLAlchemy 2.0 declarative models.

Importing this package side-effects every model into `Base.metadata`, which is
what Alembic reads for autogenerate. Anywhere you'd normally type
`from src.db.ticker import Ticker`, you can equivalently do
`from src.db import Ticker`.
"""

from src.db.base import Base, TimestampMixin
from src.db.ohlcv import OHLCV
from src.db.prediction import Prediction
from src.db.ticker import AssetType, Market, Ticker

__all__ = [
    "OHLCV",
    "AssetType",
    "Base",
    "Market",
    "Prediction",
    "Ticker",
    "TimestampMixin",
]
