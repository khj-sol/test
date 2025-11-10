import sys
import yfinance as yf
import pandas_ta as ta
import pandas as pd

def analyze_stock(ticker):
    """
    주어진 티커(주식 코드)에 대한 기술적 분석을 수행합니다.
    """
    try:
        # 1. yfinance를 통해 주식 데이터 다운로드 (최근 1년치 데이터)
        # 나스닥 티커 예: 'AAPL', 'MSFT', 'NVDA', 'GOOGL'
        data = yf.download(ticker, period="1y", interval="1d")

        if data.empty:
            print(f"오류: '{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커를 확인해 주세요.")
            return

        # 2. pandas-ta를 사용하여 보조 지표 계산
        # RSI (14일 기준)
        data.ta.rsi(length=14, append=True)
        
        # MACD (기본값: fast=12, slow=26, signal=9)
        data.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # 이동평균선 (SMA: 20일, 50일)
        data.ta.sma(length=20, append=True)
        data.ta.sma(length=50, append=True)

        # 3. 최신 데이터(오늘 또는 가장 최근 거래일) 추출
        latest_data = data.iloc[-1]

        # 4. 분석 결과 출력
        print("---" * 15)
        print(f"📊 {ticker} 기술적 분석 결과 (최근 거래일: {latest_data.name.strftime('%Y-%m-%d')})")
        print("---" * 15)
        
        print(f"종가: ${latest_data['Close']:.2f}")
        print("\n--- 주요 지표 ---")

        # RSI 분석
        rsi_14 = latest_data['RSI_14']
        print(f"RSI (14일): {rsi_14:.2f}")
        if rsi_14 > 70:
            print("  -> 📈 상태: 과매수 구간 (과열)")
        elif rsi_14 < 30:
            print("  -> 📉 상태: 과매도 구간 (침체)")
        else:
            print("  -> 📊 상태: 중립 구간")

        # 이동평균선 분석
        sma_20 = latest_data['SMA_20']
        sma_50 = latest_data['SMA_50']
        print(f"\n이동평균선 (SMA):")
        print(f"  - 20일선: ${sma_20:.2f}")
        print(f"  - 50일선: ${sma_50:.2f}")
        if sma_20 > sma_50:
            print("  -> 📈 상태: 단기 골든 크로스 (상승 추세)")
        else:
            print("  -> 📉 상태: 단기 데드 크로스 (하락 추세)")

        # MACD 분석
        macd_line = latest_data['MACD_12_26_9']
        signal_line = latest_data['MACDs_12_26_9']
        print(f"\nMACD (12, 26, 9):")
        print(f"  - MACD 선: {macd_line:.2f}")
        print(f"  - 시그널 선: {signal_line:.2f}")
        if macd_line > signal_line:
            print("  -> 📈 상태: 매수 신호 (상승 모멘텀)")
        else:
            print("  -> 📉 상태: 매도 신호 (하락 모멘텀)")

    except Exception as e:
        print(f"분석 중 오류 발생: {e}")

# --- 메인 프로그램 실행 ---
if __name__ == "__main__":
    # 명령줄에서 주식 코드를 받습니다.
    if len(sys.argv) < 2:
        print("사용법: python analyze.py [주식코드]")
        print("예시 (애플): python analyze.py AAPL")
        print("예시 (삼성전자): python analyze.py 005930.KS")
    else:
        stock_ticker = sys.argv[1].upper() # 입력받은 티커를 대문자로 변환
        analyze_stock(stock_ticker)