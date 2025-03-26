import numpy as np
import pandas as pd


def prepare_data_with_ahead_inputs(vwaps: np.ndarray, volumes: np.ndarray, features: np.ndarray, dates, lookback: int, n_ahead: int, autoscale_target: bool = True):
    n_row = vwaps.shape[0]
    normalization_section = 24 * 7 * 2
    # Preallocate arrays based on the maximum possible number of samples
    X = np.zeros((n_row - lookback - n_ahead - normalization_section, lookback + n_ahead - 1, features.shape[1]))
    y_prices = np.zeros((n_row - lookback - n_ahead - normalization_section, n_ahead, 1))
    y_volumes = np.zeros((n_row - lookback - n_ahead - normalization_section, n_ahead, 1))
    sample_dates = []  # To keep track of each sample’s forecast origin date
    
    offset = 0
    for row in range(lookback + normalization_section, n_row - n_ahead):
        idx = row - lookback - offset - normalization_section
        X[idx-offset] = features[row - lookback:row + n_ahead - 1]
        # Normalize the volume feature
        s = np.sum(features[row - lookback - normalization_section:row - 1, 0], axis=0, keepdims=True)
        if s == 0:
            offset += 1
            continue
            
        X[idx-offset, :, -1] = X[idx-offset, :, -1] / s  
        X[idx-offset, :, 0] = X[idx-offset, :, -1] / s  
        # Normalize the price feature
        
        X[idx-offset, :, -2] = X[idx-offset, :, -2] / X[idx-offset, 0:1, -2]
        if np.any(np.isinf(X[idx-offset])) or np.any(np.isnan(X[idx-offset])):
            offset += 1
            continue
        s_vol = np.sum(volumes[row - lookback:row])
        if s_vol > 0 and np.sum(volumes[row:row + n_ahead]) > 0:
            y_prices[idx-offset] = np.expand_dims(vwaps[row:row + n_ahead] / vwaps[row - lookback], axis=-1)
            y_volumes[idx-offset] = np.expand_dims(volumes[row:row + n_ahead] / s_vol, axis=-1)
            sample_dates.append(dates[row])
        else:
            offset += 1  # Skip sample if target scaling is not possible
    # Combine the target arrays along the last axis
    y = np.concatenate([y_volumes, y_prices], axis=-1)
    valid_count = len(sample_dates)
    return X[:valid_count], y[:valid_count], np.array(sample_dates)


def full_generate(volumes: pd.DataFrame, notionals: pd.DataFrame, target_asset, lookback=120, n_ahead=12, split_date=None, autoscale_target=True, include_ahead_inputs=False, freq_min=60):
    volumes = volumes.astype(np.float32).resample(f'{freq_min}min').sum()
    notionals = notionals.astype(np.float32).resample(f'{freq_min}min').sum()
    assets = [target_asset]
    
    assert target_asset in assets
    volumes = volumes[assets].dropna()
    notionals = notionals[assets].dropna()
    notionals = notionals.loc[volumes.index]
    volumes = volumes.loc[notionals.index]
    notionals.index = pd.to_datetime(notionals.index, utc=True)
    volumes.index = pd.to_datetime(volumes.index, utc=True)
    vwaps = pd.DataFrame(notionals.values / volumes.values, index=volumes.index, columns=volumes.columns)
    vwaps = vwaps.ffill().dropna()
    notionals = notionals.loc[vwaps.index]
    volumes = volumes.loc[vwaps.index]
    
    # Create features
    features = volumes #/ volumes.shift(lookback + n_ahead).rolling(24 * 7 * 2).mean()
    
    features['freq_min'] = np.float32(freq_min)
    features['hour'] = volumes.index.month
    features['dow'] = volumes.index.dayofweek
    for asset in assets:
        features[f'returns {asset}'] = vwaps[asset] / vwaps[asset].shift() - 1.
    
    # Select the target series and align
    volumes_series = volumes[target_asset]
    vwaps_series = vwaps[target_asset]
    features = features.loc[volumes_series.index].dropna()
    volumes_series = volumes_series.loc[features.index]
    vwaps_series = vwaps_series.ffill().loc[volumes_series.index]
    features['prices'] = vwaps_series.values
    features['volumes'] = volumes_series.values

    # Build samples and also capture their associated dates (forecast origin)
    X, y, sample_dates = prepare_data_with_ahead_inputs(
        vwaps_series.values, volumes_series.values, features.values, features.index, lookback, n_ahead, autoscale_target=autoscale_target
    )

    # If a split_date is provided, use it to separate train and test samples
    if split_date is not None:
        split_date = pd.to_datetime(split_date, utc=True)
        mask_train = sample_dates < split_date
        mask_test = sample_dates >= split_date
        X_train = X[mask_train]
        y_train = y[mask_train]
        X_test = X[mask_test]
        y_test = y[mask_test]
        return X_train, X_test, y_train, y_test, sample_dates
    else:
        # Fall back to size-based split if no date is provided
        test_row = int(len(y) * 0.2)
        X_train = X[:-test_row]
        X_test = X[-test_row:]
        y_train = y[:-test_row]
        y_test = y[-test_row:]
        return X_train, X_test, y_train, y_test, sample_dates

def add_config_to_X(X, minimum_traded_per_period, maximum_traded_per_period):
    if not isinstance(minimum_traded_per_period, (list, tuple)):
        minimum_traded_per_period = [minimum_traded_per_period]
    if not isinstance(maximum_traded_per_period, (list, tuple)):
        maximum_traded_per_period = [maximum_traded_per_period]
    return [
        np.concatenate([X for _ in range(len(minimum_traded_per_period))], axis=0),
        np.concatenate([np.ones((X.shape[0],1)) * v for v in minimum_traded_per_period], axis=0),
        np.concatenate([np.ones((X.shape[0],1)) * v for v in maximum_traded_per_period], axis=0),
        ]

def make_mask(shapes, minimum_period):
    mask = np.zeros(shapes)
    mask[:,:minimum_period]=True
    return mask
    
def add_config_to_y(y):
    y=np.tile(y[:,:,None,None,:], (1,1,y.shape[1]+1,4,1))
    y[:,:,:,[1,3],0] = 1.
    return y