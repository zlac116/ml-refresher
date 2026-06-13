"""
DRILL 11 — Business Calendars, Timezones, Market Hours
======================================================

OBJECTIVE
    Build a US-holiday-aware business-day calendar, shift dates by N business
    days using it, convert a NY-local timestamp to London time, and filter
    a tz-aware intraday series down to regular market hours only.

ESTIMATED TIME
    20 min

TOPICS
    pandas.tseries.holiday.USFederalHolidayCalendar
    pandas.tseries.offsets.CustomBusinessDay
    .tz_localize("US/Eastern") vs .tz_convert("Europe/London")
    Series.between_time("09:30", "16:00") — note this requires a DatetimeIndex

CANONICAL NOTE
    A bdate_range with freq=BDay treats every Mon-Fri as a business day —
    even US holidays. Use CustomBusinessDay(calendar=USFederalHolidayCalendar())
    to honour holidays.

EXPECTED OUTPUT
    US business days 2024:       251
    day after Jul 3 (skipping 4): 2024-07-05
    NYC 14:00 -> LDN HH:MM       19:00
    all-hours rows:              288
    market-only rows:            79

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def us_business_day_count(year: int) -> int:
    """Number of business days in `year` excluding US federal holidays.

    Hint: build a CustomBusinessDay calendar with USFederalHolidayCalendar()
    and pass it to pd.date_range(..., freq=cbd).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def next_us_business_day(d: pd.Timestamp) -> pd.Timestamp:
    """Add 1 business day to `d`, skipping weekends and US holidays."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def convert_nyc_to_london(naive_ts: pd.Timestamp) -> pd.Timestamp:
    """Localize `naive_ts` to US/Eastern, then convert to Europe/London."""
    # TODO: implement (hint: chain .tz_localize then .tz_convert)
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def filter_market_hours(s: pd.Series) -> pd.Series:
    """Keep only rows whose index time falls in [09:30, 16:00] inclusive.

    Use Series.between_time. The input must have a DatetimeIndex.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_bd = us_business_day_count(2024)
    assert n_bd == 251, n_bd

    after_jul3 = next_us_business_day(pd.Timestamp("2024-07-03"))
    assert after_jul3.date() == pd.Timestamp("2024-07-05").date(), after_jul3

    ldn = convert_nyc_to_london(pd.Timestamp("2024-06-13 14:00"))
    assert ldn.strftime("%H:%M") == "19:00", ldn
    assert str(ldn.tz) == "Europe/London"

    # Build a tz-aware intraday series for filtering
    np.random.seed(42)
    idx = pd.date_range("2024-01-08 00:00", "2024-01-08 23:55", freq="5min",
                        tz="US/Eastern")
    px = pd.Series(np.cumsum(np.random.normal(0, 0.1, len(idx))) + 100, index=idx)
    market = filter_market_hours(px)
    assert len(px) == 288
    assert len(market) == 79, len(market)

    print(f"US business days 2024:       {n_bd}")
    print(f"day after Jul 3 (skipping 4): {after_jul3.date()}")
    print(f"NYC 14:00 -> LDN HH:MM       {ldn.strftime('%H:%M')}")
    print(f"all-hours rows:              {len(px)}")
    print(f"market-only rows:            {len(market)}")
    print("\n✓ All checks passed.")
