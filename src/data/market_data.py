import re
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "minute_end",
    "symbol",
    "last_trade_price",
}

OPTION_SYMBOL_PATTERN = re.compile(
    r"^NIFTY"
    r"(?P<year>\d{2})"
    r"(?P<month>\d)"
    r"(?P<day>\d{2})"
    r"(?P<strike>\d{5})"
    r"(?P<option_type>CE|PE)$"
)


def parse_option_symbol(
    symbol: str,
) -> tuple[datetime, float, str]:
    """
    Extract expiry, strike and option type from a NIFTY symbol.

    Example:
    NIFTY2621025750CE

    26    -> year 2026
    2     -> February
    10    -> expiry day
    25750 -> strike
    CE    -> call option
    """

    match = OPTION_SYMBOL_PATTERN.fullmatch(symbol)

    if match is None:
        raise ValueError(
            f"Invalid NIFTY option symbol: {symbol}"
        )

    expiry = datetime(
        year=2000 + int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=15,
        minute=30,
    )

    strike = float(match.group("strike"))

    if match.group("option_type") == "CE":
        option_type = "call"
    else:
        option_type = "put"

    return expiry, strike, option_type


def load_nifty_minute_data(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load NIFTY minute-level futures and options data.

    The raw prices are stored in paise, so they are divided
    by 100 to convert them into rupees/index points.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    raw_data = pd.read_csv(
        file_path,
        dtype={
            "date": "string",
            "minute_end": "string",
            "symbol": "string",
        },
    )

    missing_columns = (
        REQUIRED_COLUMNS - set(raw_data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    raw_data["date"] = raw_data["date"].str.strip()

    raw_data["minute_end"] = (
        raw_data["minute_end"]
        .str.strip()
        .str.zfill(6)
    )

    raw_data["timestamp"] = pd.to_datetime(
        raw_data["date"] + raw_data["minute_end"],
        format="%Y%m%d%H%M%S",
        errors="raise",
    )

    raw_data["price"] = pd.to_numeric(
        raw_data["last_trade_price"],
        errors="raise",
    ) / 100.0

    if (raw_data["price"] <= 0).any():
        raise ValueError(
            "All market prices must be positive"
        )

    if raw_data.duplicated(
        ["timestamp", "symbol"]
    ).any():
        raise ValueError(
            "Duplicate timestamp-symbol observations found"
        )

    instrument_types = []
    expiries = []
    strikes = []
    option_types = []

    for symbol in raw_data["symbol"]:
        if symbol.endswith("FUT"):
            instrument_types.append("future")
            expiries.append(pd.NaT)
            strikes.append(float("nan"))
            option_types.append(None)
        else:
            (
                expiry,
                strike,
                option_type,
            ) = parse_option_symbol(symbol)

            instrument_types.append("option")
            expiries.append(expiry)
            strikes.append(strike)
            option_types.append(option_type)

    market_data = pd.DataFrame(
        {
            "timestamp": raw_data["timestamp"],
            "symbol": raw_data["symbol"],
            "instrument_type": instrument_types,
            "price": raw_data["price"],
            "expiry": expiries,
            "strike": strikes,
            "option_type": option_types,
        }
    )

    return market_data.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)