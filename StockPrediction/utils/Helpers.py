import numpy as np
from keras.callbacks import EarlyStopping

def normalize_column(column_to_normalize, columns, data):
  mean_values = []
  for column in columns:
    mean_values.append(data[column].mean())

  for col in column_to_normalize:
    # Calcola la media e la deviazione standard della colonna 'Value'
    mean_value = data[col].mean()
    std_value = data[col].std()

    # Calcola la Z-score normalization e crea una nuova colonna 'Z_Score'
    data[col] = (data[col] - mean_value) * np.mean(mean_values) / std_value

def splitting(X, y):
  # Dividi i dati in set di addestramento e test
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  train_index = int(len(X)*0.8)
  X_train = X[:train_index]
  X_test = X[train_index:]
  y_train = y[:train_index]
  y_test = y[train_index:]

  return X_train, X_test, y_train, y_test

def earlyStopping(num):
  early_stopping = EarlyStopping(monitor='val_loss', patience=num, restore_best_weights=True)
  return early_stopping