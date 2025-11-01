# src/system/Model/preprocessing/splitter.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DataSplitter:
    """
    Chronological splitter by unique dates.

    Splits the dataframe into train/val/test by *date blocks* so there is no
    look-ahead leakage. It does NOT shuffle. It assumes you will later build
    rolling windows inside each split (your SequenceDataset does that).
    """

    by: str = "date"

    def chronological_split(
        self,
        df: pd.DataFrame,
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        by: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        col = by or self.by

        out = df.copy()
        out[col] = pd.to_datetime(out[col])

        # Unique trading days in order
        dates = out[col].drop_duplicates().sort_values().to_list()
        n = len(dates)

        # Guard for tiny samples: fall back to a simple row split
        if n < 3:
            i1 = int(len(out) * 0.6)
            train = out.iloc[:i1]
            val = out.iloc[0:0]  # empty
            test = out.iloc[i1:]
            return (
                train.reset_index(drop=True),
                val.reset_index(drop=True),
                test.reset_index(drop=True),
            )

        # Compute cut points on *dates*, not rows
        i1 = int(n * ratios[0])
        i2 = i1 + int(n * ratios[1])

        # Keep cuts sane
        i1 = max(1, min(i1, n - 2))
        i2 = max(i1 + 1, min(i2, n - 1))

        t1 = dates[i1 - 1]
        t2 = dates[i2 - 1]

        train = out[out[col] <= t1]
        val = out[(out[col] > t1) & (out[col] <= t2)]
        test = out[out[col] > t2]

        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )
