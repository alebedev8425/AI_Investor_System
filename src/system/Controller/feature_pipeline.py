from __future__ import annotations

import logging
import pandas as pd

from system.Controller.cache_manager import CacheManager
from system.Model.features.technical_features import TechnicalFeatureBuilder


class FeaturePipeline:
    """
    Phase-1 controller for building technical features.

    Flow:
      * infer (price_key) from the provided prices frame (ticker set + min/max dates)
      * use CacheManager.ensure_technical(...) to reuse or (re)build deterministically
    """

    def __init__(
        self,
        *,
        cache: CacheManager,
        technical_builder: TechnicalFeatureBuilder,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cache = cache
        self._tech = technical_builder
        self._log = logger or logging.getLogger(__name__)

    def build_technical(self, prices: pd.DataFrame, *, overwrite: bool = True) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("FeaturePipeline.build_technical: received empty prices DataFrame")

        # Infer the price key from the actual frame so we don't have to change RunManager signature.
        # (Must match the key used by PriceManager: same tickers/start/end/adjust)
        tickers = sorted({str(t).upper() for t in prices["ticker"].unique()})
        start = pd.to_datetime(prices["date"]).min().date()
        end = pd.to_datetime(prices["date"]).max().date()
        # PriceManager defaulted adjust=True; keep consistent for the inferred key:
        price_key = self._cache.prices_key(tickers, start, end, adjust=True)

        feats = self._cache.ensure_technical(
            price_key=price_key,
            df_prices=prices,
            builder=self._tech,
            overwrite=overwrite,
        )
        self._log.info("Saved technical features: %d rows", len(feats))
        return feats