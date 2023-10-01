import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

target_names = ["High", "Low", "Open", "Close" ]

def plotLoss(history):
  # Plot della curva di loss
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()

# Permette di plottare un grafico zoommato su n samples dell'azienda x
def zoomPlot(num_samples, model_predictions_denormalized, y_test_denormalized, target_index, target_name, company_index, companies):
  # Seleziona le prime num_samples predizioni e valori reali
  predictions_sample = model_predictions_denormalized[-num_samples:]
  y_test_sample = y_test_denormalized[-num_samples:]

  # Crea un array di indici per l'asse x
  indices = np.arange(num_samples)

  print(indices.size)
  print(predictions_sample.size)

  # Crea il grafico
  plt.figure(figsize=(8, 4))
  plt.scatter(indices, predictions_sample[:, target_index], label='Predicted', color='blue')
  plt.plot(indices, y_test_sample[:, target_index], label='Real', color='red')
  plt.xlabel('Sample Index')
  plt.ylabel(target_name)
  plt.title(companies[company_index] + " " + target_name)
  plt.legend()
  plt.show()

def fullPlot(y_train_denormalized, y_test_denormalized, model_predictions_denormalized, target_index, target_name, company_index, dates, companies):
  plt.figure(figsize=(12, 4))
  plt.plot(dates[:len(y_train_denormalized)], y_train_denormalized[:, target_index], label='Train', color='blue')
  plt.plot(dates[-len(y_test_denormalized):], y_test_denormalized[:, target_index], label='Val', color='red')
  plt.plot(dates[-len(y_test_denormalized):], model_predictions_denormalized[:, target_index], label='Prediction', color='orange')
  plt.xlabel('Date')
  plt.ylabel(target_name + ' Price')
  plt.title('Model for ' + companies[company_index])
  day_locator = mdates.DayLocator(interval=90)  # Intervallo di 90 giorni
  plt.gca().xaxis.set_major_locator(day_locator)
  plt.legend()
  plt.show()

# plotta i 4 grafici delle predizioni di high, low, open e close in subplots
def fullPlotOfAll(y_train_denormalized, y_test_denormalized, model_predictions, company_index, dates, companies):
  print("\n")
  fig, axes = plt.subplots(2, 2, figsize=(16, 8))
  fig.suptitle("High, Low, Open, Close prediction for " + companies[company_index], fontsize=16)
  day_locator = mdates.DayLocator(interval=240)  # Intervallo di 240 giorni

  axes[0, 0].plot(dates[:len(y_train_denormalized)], y_train_denormalized[:, 0], label='Train High')
  axes[0, 0].plot(dates[-len(y_test_denormalized):], y_test_denormalized[:, 0], label='Val High', color='red')
  axes[0, 0].plot(dates[-len(y_test_denormalized):], model_predictions[:,0], label='Prediction High', color='orange')
  axes[0, 0].set_xlabel('Date')
  axes[0, 0].set_ylabel('High Price')
  axes[0, 0].xaxis.set_major_locator(day_locator)
  axes[0, 0].legend()

  axes[0, 1].plot(dates[:len(y_train_denormalized)], y_train_denormalized[:, 1], label='Train Low')
  axes[0, 1].plot(dates[-len(y_test_denormalized):], y_test_denormalized[:, 1], label='Val Low', color='red')
  axes[0, 1].plot(dates[-len(y_test_denormalized):], model_predictions[:,1], label='Prediction Low', color='orange')
  axes[0, 1].set_xlabel('Date')
  axes[0, 1].set_ylabel('Low Price')
  axes[0, 1].xaxis.set_major_locator(day_locator)
  axes[0, 1].legend()

  axes[1, 0].plot(dates[:len(y_train_denormalized)], y_train_denormalized[:, 2], label='Train Open')
  axes[1, 0].plot(dates[-len(y_test_denormalized):], y_test_denormalized[:, 2], label='Val Open', color='red')
  axes[1, 0].plot(dates[-len(y_test_denormalized):], model_predictions[:,2], label='Prediction Open', color='orange')
  axes[1, 0].set_xlabel('Date')
  axes[1, 0].set_ylabel('Open Price')
  axes[1, 0].xaxis.set_major_locator(day_locator)
  axes[1, 0].legend()

  axes[1, 1].plot(dates[:len(y_train_denormalized)], y_train_denormalized[:, 3], label='Train Close')
  axes[1, 1].plot(dates[-len(y_test_denormalized):], y_test_denormalized[:, 3], label='Test Close', color='red')
  axes[1, 1].plot(dates[-len(y_test_denormalized):], model_predictions[:,3], label='Prediction Close', color='orange')
  axes[1, 1].set_xlabel('Date')
  axes[1, 1].set_ylabel('Close Price')
  axes[1, 1].xaxis.set_major_locator(day_locator)
  axes[1, 1].legend()

  plt.show()
