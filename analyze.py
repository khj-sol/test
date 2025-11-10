"""Command line tool for basic technical analysis of equities using Yahoo Finance data."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd
import pandas_ta as ta
import yfinance as yf


RSI_COLUMN = "RSI_14"
SMA20_COLUMN = "SMA_20"
SMA50_COLUMN = "SMA_50"
MACD_COLUMN = "MACD_12_26_9"
MACD_SIGNAL_COLUMN = "MACDs_12_26_9"


def download_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download historical stock data for the given ticker.

    Parameters
    ----------
    ticker: str
        Stock symbol accepted by Yahoo Finance.
    period: str
        History length (for example ``"1y"`` or ``"6mo"``).
    interval: str
        Sample interval, such as ``"1d"`` or ``"1h"``.

    Returns
    -------
    pandas.DataFrame
        Downloaded OHLCV data.

    Raises
    ------
    ValueError
        If Yahoo Finance returns an empty dataset.
    """

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(
            f"'{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커와 기간/간격을 확인해 주세요."
        )
    return data


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Append technical indicators to the provided dataframe."""

    data = data.copy()
    data.ta.rsi(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    data.ta.sma(length=20, append=True)
    data.ta.sma(length=50, append=True)
    return data


def format_float(value: float) -> str:
    """Format a float value consistently for CLI output."""

    return f"{value:.2f}"


def analyze_stock(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    export_path: Optional[str] = None,
) -> bool:
    """Perform technical analysis for the given ticker.

    Returns ``True`` if the analysis succeeded; otherwise ``False``.
    """

    try:
        data = download_stock_data(ticker, period, interval)
    except ValueError as error:
        print(f"오류: {error}")
        return False
    except Exception as error:  # pragma: no cover - defensive logging
        print(f"데이터 다운로드 중 예상치 못한 오류 발생: {error}")
        return False

    try:
        enriched_data = compute_indicators(data)
    except Exception as error:  # pragma: no cover - defensive logging
        print(f"보조 지표 계산 중 오류 발생: {error}")
        return False

    latest_data = enriched_data.iloc[-1]

    print("---" * 15)
    print(
        f"📊 {ticker} 기술적 분석 결과 (최근 거래일: "
        f"{latest_data.name.strftime('%Y-%m-%d')})"
    )
    print("---" * 15)

    print(f"종가: ${format_float(latest_data['Close'])}")
    print("\n--- 주요 지표 ---")

    rsi_14 = latest_data.get(RSI_COLUMN)
    if pd.notna(rsi_14):
        print(f"RSI (14일): {format_float(rsi_14)}")
        if rsi_14 > 70:
            print("  -> 📈 상태: 과매수 구간 (과열)")
        elif rsi_14 < 30:
            print("  -> 📉 상태: 과매도 구간 (침체)")
        else:
            print("  -> 📊 상태: 중립 구간")
    else:
        print("RSI (14일): 계산되지 않았습니다.")

    sma_20 = latest_data.get(SMA20_COLUMN)
    sma_50 = latest_data.get(SMA50_COLUMN)
    print("\n이동평균선 (SMA):")
    if pd.notna(sma_20):
        print(f"  - 20일선: ${format_float(sma_20)}")
    else:
        print("  - 20일선: 계산되지 않았습니다.")
    if pd.notna(sma_50):
        print(f"  - 50일선: ${format_float(sma_50)}")
    else:
        print("  - 50일선: 계산되지 않았습니다.")
    if pd.notna(sma_20) and pd.notna(sma_50):
        if sma_20 > sma_50:
            print("  -> 📈 상태: 단기 골든 크로스 (상승 추세)")
        else:
            print("  -> 📉 상태: 단기 데드 크로스 (하락 추세)")

    macd_line = latest_data.get(MACD_COLUMN)
    signal_line = latest_data.get(MACD_SIGNAL_COLUMN)
    print("\nMACD (12, 26, 9):")
    if pd.notna(macd_line):
        print(f"  - MACD 선: {format_float(macd_line)}")
    else:
        print("  - MACD 선: 계산되지 않았습니다.")
    if pd.notna(signal_line):
        print(f"  - 시그널 선: {format_float(signal_line)}")
    else:
        print("  - 시그널 선: 계산되지 않았습니다.")
    if pd.notna(macd_line) and pd.notna(signal_line):
        if macd_line > signal_line:
            print("  -> 📈 상태: 매수 신호 (상승 모멘텀)")
        else:
            print("  -> 📉 상태: 매도 신호 (하락 모멘텀)")

    if export_path:
        try:
            enriched_data.to_csv(export_path)
            print(f"\n데이터가 '{export_path}' 파일로 저장되었습니다.")
        except Exception as error:  # pragma: no cover - file system issues
            print(f"CSV 저장 중 오류 발생: {error}")

    return True


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the CLI interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Yahoo Finance 데이터를 이용해 주식의 주요 기술 지표를 계산하고 출력합니다."
        )
    )
    parser.add_argument("ticker", help="분석할 주식 코드 (예: AAPL, 005930.KS)")
    parser.add_argument(
        "--period",
        default="1y",
        help="데이터 기간 (예: 1mo, 6mo, 1y, 5y). 기본값은 1y 입니다.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="데이터 간격 (예: 1d, 1h, 30m). 기본값은 1d 입니다.",
    )
    parser.add_argument(
        "--export",
        help="보조 지표가 포함된 전체 데이터를 CSV 파일로 저장합니다.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point used by the command line script."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    success = analyze_stock(
        args.ticker.upper(), period=args.period, interval=args.interval, export_path=args.export
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
