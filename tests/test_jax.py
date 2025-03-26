import os
import tempfile
BACKEND = 'jax'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import numpy as np
import pandas as pd
import keras

from lems.loss import lem_loss
from lems.metrics import (
    buy_vwap_avg_imp,
    buy_twap_avg_imp,
    sell_vwap_avg_imp,
    sell_twap_avg_imp,
    buy_vwap_avg_risk,
    buy_twap_avg_risk,
    sell_vwap_avg_risk,
    sell_twap_avg_risk,
    unsigned_vwap_avg_risk,
    unsigned_twap_avg_risk
)
from lems.data_formater import full_generate, add_config_to_X, add_config_to_y, prepare_data_with_ahead_inputs
from lems.models import LargeExecutionModel


@pytest.fixture
def model_parameters():
    return {
        'batch_size': 64,
        'epochs': 3,
        'lookback': 15,
        'sig_lookback': 30,
        'n_ahead': 6,
        'target_asset': 'AAPL'
    }

@pytest.fixture
def generated_data_nonsig(model_parameters):
    def generate_random_data(asset_list, num_periods):
        end_date = pd.Timestamp.now().floor('h')
        start_date = end_date - pd.Timedelta(hours=num_periods-1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        data = np.exp(np.random.rand(num_periods, len(asset_list)))
        df = pd.DataFrame(data, columns=asset_list, index=date_range)
        return df

    notionals = generate_random_data([model_parameters['target_asset']], 1000)
    volumes = generate_random_data([model_parameters['target_asset']], 1000)
    
    
    X_train, X_test, y_train, y_test, sample_dates = full_generate(
        volumes, 
        notionals,
        model_parameters['target_asset'],
        lookback=model_parameters['lookback'],
        n_ahead=model_parameters['n_ahead'],
    )

    
    
    return add_config_to_X(X_train, 0., 1.), add_config_to_X(X_test, 0., 1.), add_config_to_y(y_train), add_config_to_y(y_test)


@pytest.fixture(params=['adam', 'sgd'])
def optimizer(request):
    return request.param

def test_lem_fit_and_save(model_parameters, generated_data_nonsig, optimizer):
    assert keras.backend.backend() == BACKEND
    
    X_train, X_test, y_train, y_test = generated_data_nonsig
    
    model = LargeExecutionModel(
        lookback=model_parameters['lookback'],
        n_ahead=model_parameters['n_ahead'],
        num_embedding=4,
        num_heads=2,
        hidden_size=50,
        fused_mlp_hidden_dim=20,
    )
    
    model.compile(optimizer=keras.optimizers.Adam(0.00001),
                                   loss=lem_loss,
                                   metrics=[
                                       buy_vwap_avg_imp,
                                       buy_twap_avg_imp,
                                       sell_vwap_avg_imp,
                                       sell_twap_avg_imp,
                                       buy_vwap_avg_risk,
                                       buy_twap_avg_risk,
                                       sell_vwap_avg_risk,
                                       sell_twap_avg_risk,
                                       unsigned_vwap_avg_risk,
                                       unsigned_twap_avg_risk,
                                   ],
                                )

    
    history = model.fit(
        X_train, y_train,
        batch_size=model_parameters['batch_size'],
        epochs=model_parameters['epochs'],
        validation_split=0.2,
        shuffle=True,
        verbose=False
    )
    
    # Get predictions before saving
    predictions_before = model.predict(X_test, verbose=False)
    
    # Save and load the model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'dynamic_model.keras')
        model.save(model_path)
        loaded_model = keras.models.load_model(model_path, compile=False)
    
    # Get predictions after loading
    predictions_after = loaded_model.predict(X_test, verbose=False)
    
    # Compare predictions
    np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-8)
    
    print(f"dynamic transformer VWAP model with {optimizer} optimizer successfully saved, loaded, and reused.")

