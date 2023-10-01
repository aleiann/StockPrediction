from attention import MultiHeaded
from keras import layers, models, regularizers

def bert_like(X_train, n_hidden_units, n_heads):
       n_hidden_units = n_hidden_units
       n_heads = n_heads
       dim = 64
       # Creazione del modello
       model0 = models.Sequential([
           layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
           layers.Dropout(0.03),  # Aggiunta di dropout come regolarizzazione
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu'),
           layers.Dropout(0.03)
       ])
       model1 = models.Sequential([
           #layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
           layers.Dropout(0.03),  # Aggiunta di dropout come regolarizzazione
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu'),
           layers.Dropout(0.03)
       ])
       model2 = models.Sequential([
           #layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
           layers.Dropout(0.03),  # Aggiunta di dropout come regolarizzazione
           # SelfAttention.SelfAttention(embed_dim=64),
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu', kernel_regularizer=regularizers.l2(0.002))
       ])
       model3 = models.Sequential([
           #layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
           layers.Dropout(0.03),  # Aggiunta di dropout come regolarizzazione
           # SelfAttention.SelfAttention(embed_dim=64),
           MultiHeaded.MultiHeadedAttention(num_heads=n_heads, head_dim=dim),
           layers.Dense(n_hidden_units, activation='relu', kernel_regularizer=regularizers.l2(0.001))
       ])
       model = models.Sequential()
       model.add(model0)
       model.add(model1)
       model.add(model2)
       model.add(model3)
       model.add(layers.GlobalAveragePooling1D())
       model.add(layers.Dense(5, activation='relu'))
       return model

"""
semplified gridSerach
"""
"""
def gridSearch(n_hidden_units, n_heads, X_train, y_train, X_test, y_test, optimizers, epochs):
    combos = list(itertools.product(n_hidden_units, n_heads, optimizers, epochs))
    losses = []
    for combo in combos:
        model = bert_like(X_train, combo[0], combo[1])
        model.compile(optimizer=combo[2](learning_rate=1e-3),
                      loss=ll.mean_squared_error)
        # Addestra il modello con approccio supervised
        model.fit(X_train, y_train, epochs=combo[3], batch_size=32, callbacks=[help.earlyStopping(10)],
                            validation_split=0.2)

        # Valuta il modello
        loss = model.evaluate(X_test, y_test)
        losses.append(loss)
    best_mse = min(losses)
    index_best_params = losses.index(best_mse)
    best_params = combos[index_best_params]

    return best_params, best_mse
"""








