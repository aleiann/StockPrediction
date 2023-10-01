from utils import Helpers as help
from models import BiLSTMModel as modelB
from models import HybridModel as modelH
from preProcessing import PreProcess
from graphicFunction import Functions as gf
from keras.optimizers import Adam, AdamW, Lion
from loss import Losses as ll
from keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError
from metrics import Metrics as mt
from models import BertLike as es


#---------------------------PREPROCESS----------------------------------------------------------------------

companies = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
company_index = 3
X_train, X_test, y_train, y_test, dates, scaler = PreProcess.prep(companies, company_index)
target_names = ["High", "Low", "Open", "Close"]

#----------------------------BERT-LIKE-----------------------------------------------------------------------

model = es.bert_like(X_train, 128, 6)
model.compile(optimizer=Adam(learning_rate=0.0003),
            loss=ll.mean_squared_error,
            metrics=[mt.accuracy, MeanAbsolutePercentageError(), RootMeanSquaredError()])

history = model.fit(X_train, y_train, epochs=150, batch_size=32, callbacks=[help.earlyStopping(10)], validation_split=0.2)

# Valuta il modello
loss = model.evaluate(X_test, y_test)

# Esegui predizioni
predictionsBert = model.predict(X_test)

# Denormalizza le predizioni
predictionsBert_denormalized = scaler.inverse_transform(predictionsBert)
y_test_denormalized = scaler.inverse_transform(y_test)
y_train_denormalized = scaler.inverse_transform(y_train)

print(f'Loss sul set di test: {loss}')

#gf.plotLoss(history)

# stampa i grafici completi dei 4 targets
gf.fullPlotOfAll(y_train_denormalized, y_test_denormalized, predictionsBert_denormalized, company_index, dates, companies)

#-----------------------------------------BILSTM-------------------------------------------------------------------

modelBiLSTM = modelB.modelBiLSTM(X_train)

modelBiLSTM.compile(optimizer=Adam(learning_rate=0.0003),
                    loss=ll.mean_squared_error,
                    metrics=[mt.accuracy, MeanAbsolutePercentageError(), RootMeanSquaredError()])

history = modelBiLSTM.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[help.earlyStopping(10)], validation_split=0.2)

# Valuta il modello
loss = modelBiLSTM.evaluate(X_test, y_test)

# Esegui predizioni
predictionsBiLSTM = modelBiLSTM.predict(X_test)

# Denormalizza le predizioni
predictions_BiLSTM_denormalized = scaler.inverse_transform(predictionsBiLSTM)
y_test_denormalized = scaler.inverse_transform(y_test)
y_train_denormalized = scaler.inverse_transform(y_train)

print(f'Loss sul set di test: {loss}')

gf.plotLoss(history)

# stampa i grafici completi dei 4 targets
gf.fullPlotOfAll(y_train_denormalized, y_test_denormalized, predictions_BiLSTM_denormalized, company_index, dates, companies)


#--------------------HYBRID MODEL----------------------------------------------------

model = modelH.hybrid(X_train, 128, 6)
model.compile(optimizer=Adam(learning_rate=0.0003),
                loss=ll.mean_squared_error,
                metrics=[mt.accuracy, MeanAbsolutePercentageError(), RootMeanSquaredError()])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[help.earlyStopping(10)], validation_split=0.2)
# Valuta il modello
loss = model.evaluate(X_test, y_test)

"""
codice per stampare in tabella i risultati
 mse_lf = round(loss[0], 4)
    accuracy_percentage = round(loss[1] * 100, 2)
    mape = round(loss[2], 2)
    rmse = round(loss[3], 3)
    new_row = [int(i), mse_lf, accuracy_percentage, mape, rmse]
    results.loc[len(results)] = new_row
#Table.printTable(results, 'HYBRID MODEL')
"""

predictionsHybrid = model.predict(X_test)

# Denormalizza le predizioni
predictionsHybrid_denormalized = scaler.inverse_transform(predictionsHybrid)
y_test_denormalized = scaler.inverse_transform(y_test)
y_train_denormalized = scaler.inverse_transform(y_train)

print(f'Loss sul set di test: {loss}')
gf.plotLoss(history)

# stampa i grafici completi dei 4 targets
gf.fullPlotOfAll(y_train_denormalized, y_test_denormalized, predictionsHybrid_denormalized, company_index, dates, companies)

