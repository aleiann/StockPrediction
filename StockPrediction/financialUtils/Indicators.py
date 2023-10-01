import numpy as np
import pandas as pd

#calcolo buy & sell volume
def getBuySellVolume(data):
  # Calcolo del cambiamento giornaliero dei prezzi
  data['Price variation'] = data['Close'] - data['Close'].shift(1)

  # Calcola la differenza tra il prezzo di chiusura e apertura per ogni giorno
  data['Price_Difference'] = data['Close'] - data['Open']

  # Calcola la differenza tra il prezzo massimo e il prezzo minimo
  data['Price_Range'] = data['High'] - data['Low']

  # Stima se il giorno è di acquisto (+1) o vendita (-1) in base alla Price_Difference
  data['Buy_Sell'] = data['Price_Difference'].apply(lambda x: 1 if x > 0 else -1)

  # Calcola un indice di attività basato sulla Price_Range e il volume
  data['Activity_Index'] = data['Price_Range'] * data['Volume']

  # Crea una colonna di Volume di Acquisto moltiplicando l'indice di attività per il segno di acquisto
  data['Buy_Volume'] = data['Activity_Index'] * data['Buy_Sell']

  # Crea una colonna di Volume di Vendita moltiplicando l'indice di attività per il segno di vendita (invertito rispetto all'acquisto)
  data['Sell_Volume'] = data['Activity_Index'] * -data['Buy_Sell']

# Calcolo del RSI
def calculate_rsi(data, window=14):
    delta = data['Price variation']
    gain = np.where(delta > 0, delta, 0) #gain data con 0 dove delta < 0
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcolo della Media Mobile Esponenziale (EMA)
def calculate_ema(data, window=14):
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    return ema

# Calcolo del market sentiment
def getMarketSentiment(rsi, ema, volume):
  return (0.5 * rsi) + (0.4 * ema) + (0.1 * volume)