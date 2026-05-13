import pandas as pd
import polars as pl
import pytest

from antupy import Var
from antupy.tsg.settings import TimeParams


def test_days_and_periods():
    tp = TimeParams(
        START=Var(0, "hr"),
        STOP=Var(48, "hr"),
        STEP=Var(30, "min"),
        YEAR=Var(2024, "-"),
    )

    assert tp.DAYS == Var(2, "day")
    assert tp.PERIODS == Var(96, "-")


def test_idx_polars_default_engine():
    tp = TimeParams(
        START=Var(0, "hr"),
        STOP=Var(24, "hr"),
        STEP=Var(60, "min"),
        YEAR=Var(2024, "-"),
    )

    idx = tp.idx
    assert isinstance(idx, pl.Series)
    assert len(idx) == int(tp.PERIODS.gv("-"))


def test_idx_pandas_engine():
    tp = TimeParams(
        START=Var(1, "hr"),
        STOP=Var(25, "hr"),
        STEP=Var(60, "min"),
        YEAR=Var(2024, "-"),
        engine="pandas",
    )

    idx = tp.idx
    assert isinstance(idx, pd.DatetimeIndex)
    assert len(idx) == int(tp.PERIODS.gv("-"))
    assert idx[0] == pd.Timestamp("2024-01-01 01:00:00")


def test_idx_invalid_engine_raises():
    tp = TimeParams(engine="bad_engine")

    with pytest.raises(ValueError, match="engine must be 'polars' or 'pandas'"):
        _ = tp.idx


def test_idx_pd_frequency_and_start():
    tp = TimeParams(
        START=Var(2, "hr"),
        STOP=Var(8, "hr"),
        STEP=Var(30, "min"),
        YEAR=Var(2025, "-"),
    )

    idx = tp.idx_pd
    assert idx[0] == pd.Timestamp("2025-01-01 02:00:00")
    assert len(idx) == int(tp.PERIODS.gv("-"))
    assert idx[1] - idx[0] == pd.Timedelta(minutes=30)
