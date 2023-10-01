
from keras import layers, models, regularizers
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.regularizers import L2
from attention import SelfAttention, MultiHeaded

def hybrid(X_train, n_hidden_units, n_heads):
       n_hidden_units = n_hidden_units
       n_heads = n_heads
       dim = 64
       # Creazione del modello
       model0 = models.Sequential([
           layers.Bidirectional(LSTM(units=128, return_sequences=True, kernel_regularizer=L2(0.005),
                                     input_shape=(X_train.shape[1], X_train.shape[2]))),
           layers.Dropout(0.03),  # Aggiunta di dropout come regolarizzazione
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dropout(0.2),
           layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
           layers.Dropout(0.03)
       ])

       model1 = models.Sequential([
           layers.Dropout(0.03),
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu'),
           layers.Dropout(0.03)
       ])
       model2 = models.Sequential([
           layers.Dropout(0.03),
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu', kernel_regularizer=regularizers.l2(0.002))
       ])
       model3 = models.Sequential([
           layers.Dropout(0.03),
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu'),
           layers.Dropout(0.03)
       ])
       model = models.Sequential()
       model.add(model0)
       model.add(model1)
       model.add(model2)
       model.add(model3)
       model.add(layers.GlobalAveragePooling1D())
       model.add(layers.Dense(5, activation='relu'))
       return model

