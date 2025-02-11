import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def meta():
    df = pd.read_csv("tests/__data__/meta.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    return df.set_index("Date").Close
