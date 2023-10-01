import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Flatten
from keras.regularizers import L2
from attention import SelfAttention, MultiHeaded

def modelBiLSTM(X_train):
    # Crea il modello
    modelBiLSTM = Sequential()
    modelBiLSTM.add(Bidirectional(LSTM(units=128, return_sequences=True, kernel_regularizer=L2(0.005), input_shape=(X_train.shape[1], X_train.shape[2]))))
    selfAttentionLayer = SelfAttention.SelfAttention(embed_dim=64)
    modelBiLSTM.add(selfAttentionLayer)
    modelBiLSTM.add(Flatten())
    modelBiLSTM.add(Dense(units=128, activation='relu'))
    modelBiLSTM.add(Dropout(0.2))
    modelBiLSTM.add(Dense(units=128, activation='relu'))
    modelBiLSTM.add(Dropout(0.3))
    modelBiLSTM.add(Dense(units=5, activation='relu'))  # 5 uscite: [High, Low, Open, Close],'Volume'
    return modelBiLSTM



