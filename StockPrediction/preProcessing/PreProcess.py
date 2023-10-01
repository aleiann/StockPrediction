from datetime import timedelta
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from financialUtils import Indicators as ind
from utils import Helpers as help
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import altair as alt
from statsmodels.tsa.api import SimpleExpSmoothing


def prep(companies, company_index):
    # Carico i dati
    data = pd.read_csv(companies[company_index] + '.csv')
    data = setNanValues(data)
    # Seleziono i dati rilevanti
    data = data[['Date','High', 'Low', 'Open', 'Close','Volume']]
    ind.getBuySellVolume(data)
    data['RSI'] = ind.calculate_rsi(data)
    data['EMA'] = ind.calculate_ema(data)

    # Rimuovi le righe con valori NaN dovuti al calcolo dell'RSI
    data = data.dropna()

    dates = data['Date'].values

    columns = ["High", "Low", "Open", "Close" ]
    column_to_normalize = ['Buy_Volume']
    help.normalize_column(column_to_normalize, columns, data)

    # Calcola una media ponderata di RSI ed EMA (puoi regolare i pesi in base alle tue preferenze)
    data['Market_Sentiment'] = ind.getMarketSentiment(data['RSI'],data['EMA'],data['Buy_Volume'])

    data = data[['Date','High', 'Low', 'Open', 'Close', 'Market_Sentiment']]
    data1 = data.set_index('Date')
    dataHigh = data1[['High']]
    anomaly_datesR = anomalyResid(dataHigh, company_index)
    anomaly_datesIF = isolationForest(dataHigh)
    valori_none = [None] * len(data1.columns)
    start_date = dates.min()
    end_date = dates.max()

    #setta i valori None per le anomalie individuate
    for date in anomaly_datesIF:
        if date.strftime('%Y-%m-%d') != start_date and date.strftime('%Y-%m-%d') != end_date:
            data1.loc[date.strftime('%Y-%m-%d')] = valori_none

    for date in anomaly_datesR:
        if date.strftime('%Y-%m-%d') != start_date and date.strftime('%Y-%m-%d') != end_date:
            data1.loc[date.strftime('%Y-%m-%d')] = valori_none

    #interpolazione per anomalie
    data1 = data1.interpolate()

    print(data.head())
    dataset = data1.values

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    X, y = [], []
    look_back = 3  # Lunghezza della sequenza passata da utilizzare per la previsione

    for i in range(len(dataset_scaled) - look_back):
        X.append(dataset_scaled[i:i+look_back])
        y.append(dataset_scaled[i+look_back])

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = help.splitting(X,y)

    return X_train, X_test, y_train, y_test, dates, scaler

def setNanValues(df):
    # Crea un DataFrame con tutte le date desiderate
    dates = df['Date'].values
    start_date = dates.min()
    end_date = dates.max()
    date_range = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')

    for date in date_range:
        if date not in dates and date != end_date:
            index = (datetime.datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            i = np.where(dates == index)[0][0]
            high = (df.at[i, 'High'] + df.at[i+1, 'High'])/2
            low = (df.at[i, 'Low'] + df.at[i + 1, 'Low']) / 2
            openV = (df.at[i, 'Open'] + df.at[i + 1, 'Open']) / 2
            close = (df.at[i, 'Close'] + df.at[i + 1, 'Close']) / 2
            volume = (df.at[i, 'Volume'] + df.at[i + 1, 'Volume']) / 2
            new_row = pd.DataFrame({'Date' : [date],
                                    'High': [high],
                                    'Low': [low],
                                    'Open': [openV],
                                    'Close': [close],
                                    'Volume': [volume]},)
            df = pd.concat([df[:i+1], new_row, df[i+1:]])
            lenght = len(df.index)
            indexes = list(range(lenght))
            df = df.set_index(pd.Index(indexes))
            dates = np.insert(dates, i+1, date)

    # Riordina l'indice
    df = df.sort_index()

    # Resetta l'indice
    df = df.reset_index()

    return df

def anomalyResid(df, index):
    # infer the frequency of the data
    df.index = pd.to_datetime(df.index)
    start_date = datetime.datetime(2017, 1, 7)
    end_date = datetime.datetime(2017, 12, 31)
    #df = df[start_date:end_date]
    plt.rc('figure', figsize=(12, 8))
    plt.rc('font', size=15)

    result = seasonal_decompose(df, model='additive')
    fig = result.plot()
    fig.show()

    plt.rc('figure', figsize=(12, 6))
    plt.rc('font', size=15)
    soglia = 0

    #soglie individuate per le 4 aziende
    match index:
        case 0:
            soglia = 3;
        case 1:
            soglia = 4;
        case 2:
            soglia = 5;
        case 3:
            soglia = 2;

    fig, ax = plt.subplots()
    x = result.resid.index
    y = result.resid.values
    anomaly_dates = []
    anomalies = []
    ax.plot_date(x, y, color='black', linestyle='--')
    ax.axhline(y=soglia, color='blue', linestyle='--', label='Soglia')
    ax.axhline(y=-soglia, color='blue', linestyle='--', label='Soglia')
    # Aggiungi annotazioni quando il valore supera la soglia
    for i in range(len(y)):
        if np.abs(y[i]) > soglia:
            ax.annotate('Anomaly', (x[i], y[i]), xytext=(30, 20), textcoords='offset points',
                        color='red', arrowprops=dict(facecolor='red', arrowstyle='fancy'))
            anomaly = df.iloc[i]
            anomalies.append(anomaly)
            anomaly_dates.append(anomaly.name)

    anomalies = pd.DataFrame(anomalies)

    fig.autofmt_xdate()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))  # anomaly

    ax.plot(df.index, df['High'], color='black', label='Normal')
    ax.scatter(anomalies.index, anomalies['High'], color='red', label='Anomaly')
    plt.legend()
    plt.title("Resid Anomaly Detection")
    plt.show();

    return anomaly_dates

def isolationForest(df):
    outliers_fraction = float(.02)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df.values.reshape(-1, 1))
    data = pd.DataFrame(np_scaled)

    # train isolation forest
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)

    df['anomaly'] = model.predict(data)

    # visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    a = df.loc[df['anomaly'] == -1, ['High']]  # anomaly

    ax.plot(df.index, df['High'], color='black', label='Normal')
    ax.scatter(a.index, a['High'], color='red', label='Anomaly')
    plt.legend()
    plt.title("Isolation Forest Anomaly Detection")
    plt.show();

    lenght = len(a)
    anomaly_dates = []
    for i in range(0,lenght):
        anomaly_dates.append(a.iloc[i].name)

    return anomaly_dates





