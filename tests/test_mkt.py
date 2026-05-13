import json

import pandas as pd
import pytest

import antupy.tsg.mkt as mkt
from antupy.tsg.mkt import MarketAU, MarketCL


def test_market_au_load_data_with_provided_file(monkeypatch):
    idx = pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01"])
    df_sp = pd.DataFrame(
        {
            "SP_NSW": [10.0, 20.0, 30.0],
            "Demand_NSW": [100.0, 110.0, 120.0],
        },
        index=idx,
    )

    monkeypatch.setattr(pd, "read_pickle", lambda _: df_sp)

    market = MarketAU(state="NSW", year_i=2019, year_f=2019, file_data="dummy.pkl")
    out = market.load_data()

    assert list(out.columns) == ["spot_price"]
    assert len(out) == 1
    assert out.iloc[0]["spot_price"] == 20.0


def test_market_cl_load_data_dispatches_json(monkeypatch):
    expected = pd.DataFrame({"spot_price": [1.0]})
    market = MarketCL(file_data="market.tsv")

    monkeypatch.setattr(MarketCL, "_load_json_format", lambda self: expected)

    out = market.load_data()
    pd.testing.assert_frame_equal(out, expected)


def test_market_cl_load_data_dispatches_csv(monkeypatch):
    expected = pd.DataFrame({"spot_price": [2.0]})
    market = MarketCL(file_data="market.csv")

    monkeypatch.setattr(MarketCL, "_load_csv_format", lambda self: expected)

    out = market.load_data()
    pd.testing.assert_frame_equal(out, expected)


def test_market_cl_load_data_missing_files_raises(monkeypatch):
    monkeypatch.setattr(mkt, "DIR_DATA", "base_data", raising=False)
    monkeypatch.setattr(mkt.os.path, "isfile", lambda _: False)

    market = MarketCL(file_data=None)

    with pytest.raises(FileNotFoundError):
        market.load_data()


def test_market_cl_load_json_format(tmp_path):
    payload = [
        ["2024-01-01 00:00:00", "Crucero", 100.0],
        ["2024-01-01 01:00:00", "crucero", 120.0],
        ["2025-01-01 00:00:00", "crucero", 999.0],
    ]
    file_json = tmp_path / "market.tsv"
    file_json.write_text(json.dumps(payload), encoding="utf-8")

    market = MarketCL(location="crucero", year_i=2024, year_f=2024, dT=1.0, file_data=str(file_json))
    out = market._load_json_format()

    assert "spot_price" in out.columns
    assert len(out) == 2
    assert out.index.tz is not None


def test_market_cl_load_csv_format(tmp_path):
    # index_col=3 in implementation, so date must be 4th column.
    file_csv = tmp_path / "market.csv"
    file_csv.write_text(
        "fecha,hora,cmg,date\n"
        "2024-01-01,00:00,100,2024-01-01 00:00:00\n"
        "2024-01-01,01:00,110,2024-01-01 01:00:00\n"
        "2025-01-01,00:00,999,2025-01-01 00:00:00\n",
        encoding="utf-8",
    )

    market = MarketCL(year_i=2024, year_f=2024, file_data=str(file_csv))
    out = market._load_csv_format()

    assert len(out) == 2
    assert out.index.tz is not None
