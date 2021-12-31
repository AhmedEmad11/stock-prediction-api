from rest_framework.decorators import api_view
from rest_framework.response import Response

import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime as dt
import pandas_datareader as web


@api_view(['GET'])
def facebook(request):
    mdl = joblib.load('ai/stock_model.pkl')
    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()
    prediction_days = 60

    company = 'FB'

    scaler = MinMaxScaler(feature_range=(0, 1))
    start = dt.datetime(2012,1,1)
    end = dt.datetime(2020,1,1)
    data = web.DataReader(company, 'yahoo', start, end)

    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset)- len(test_data) - prediction_days :].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    real_data = [model_inputs[len(model_inputs + 1)  - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = mdl.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    res = {
        'Prediction': prediction[0][0]
    }
    return Response(res)
    
