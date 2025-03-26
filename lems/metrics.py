import keras
from keras import ops

def create_mask(batch, n_ahead):
    base_mask = ops.tile(ops.transpose(ops.tril(ops.ones((n_ahead, n_ahead)), k=0))[None,:,:,None,None], (batch, 1, 1, 4, 2))
    return ops.concatenate([base_mask, ops.ones((batch, n_ahead, 1, 4, 2))], axis = 2)

def buy_vwap_avg_imp(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )

    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0

    pnl = -diffs[:,:-1,0,:]
    return ops.mean(pnl)


def buy_twap_avg_imp(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )

    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0

    pnl = -diffs[:,:-1,1,:]
    return ops.mean(pnl)

def sell_vwap_avg_imp(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0

    pnl = diffs[:,:-1,2,:]
    return ops.mean(pnl)

def sell_twap_avg_imp(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )

    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0

    pnl = diffs[:,:-1,3,:]
    return ops.mean(pnl)

def buy_vwap_avg_risk(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.)
    risk_diff = keras.ops.maximum(diffs[:,:-1,0,:], 0.)
    vwap_mkt = keras.ops.sum(mkt_volumes[:,:,0,0] * mkt_prices[:,:,0,0], axis=1) / \
               keras.ops.sum(mkt_volumes[:,:,0,0], axis=1)

    twap = keras.ops.sum(mkt_prices[:,:,0,0], axis=1) / y_true.shape[1]
    twap_vwap_diff = twap / vwap_mkt - 1.
    baseline_risk = keras.ops.maximum(twap_vwap_diff, 0.)
    loss = keras.ops.mean(risk_diff) / (keras.ops.mean(baseline_risk) + eps) - 1.
    return loss * 100.


def buy_twap_avg_risk(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0
    risk_diff = keras.ops.abs(diffs[:,:-1,1,:])
    loss = keras.ops.mean(risk_diff)
    return loss * 100.


def sell_vwap_avg_risk(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.)
    risk_diff = keras.ops.maximum(-diffs[:,:-1,2,:], 0.)
    vwap_mkt = keras.ops.sum(mkt_volumes[:,:,0,0] * mkt_prices[:,:,0,0], axis=1) / \
               keras.ops.sum(mkt_volumes[:,:,0,0], axis=1)

    twap = keras.ops.sum(mkt_prices[:,:,0,0], axis=1) / y_true.shape[1]
    twap_vwap_diff = twap / vwap_mkt - 1.
    baseline_risk = keras.ops.maximum(-twap_vwap_diff, 0.)
    loss = keras.ops.mean(risk_diff) / (keras.ops.mean(baseline_risk) + eps) - 1.
    return loss * 100.

def sell_twap_avg_risk(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0
    risk_diff = keras.ops.abs(diffs[:,:-1,3,:])
    loss = keras.ops.mean(risk_diff)
    return loss * 100.


def unsigned_vwap_avg_risk(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.)
    risk_diff = (keras.ops.abs(diffs[:,-1:,0,:]) + keras.ops.abs(diffs[:,-1:,2,:]))/2.
    vwap_mkt = keras.ops.sum(mkt_volumes[:,:,0,0] * mkt_prices[:,:,0,0], axis=1) / \
               keras.ops.sum(mkt_volumes[:,:,0,0], axis=1)

    twap = keras.ops.sum(mkt_prices[:,:,0,0], axis=1) / y_true.shape[1]
    twap_vwap_diff = twap / vwap_mkt - 1.
    baseline_risk = keras.ops.abs(twap_vwap_diff)
    loss = keras.ops.mean(risk_diff) / (keras.ops.mean(baseline_risk) + eps) - 1.
    return loss * 100.

def unsigned_twap_avg_risk(y_true, y_pred):
    """
    y_true: (batch, n_ahead, n_ahead + 1, 4, 2) => [market_volume, price] on last dimension, repeated on n_ahead, 4 others, first n_ahead
    being the allocation period and second being min period
    y_pred: (batch, n_ahead, n_ahead + 1, 4, 2) => [buy_vwap, buy_twap, sell_vwap, sell_twap] * [volume target, notional target]
    """
    mkt_volumes = y_true[...,0:1] 
    mkt_prices = y_true[...,1:2]
    # External (forced) mask (assumed provided as 0/1) - represent minimum execution time
    mask = ops.cast(create_mask(ops.shape(y_pred)[0], ops.shape(y_pred)[1]), dtype=y_true.dtype)

    mkt_minimum_trade_mask = ops.tile(
        ops.concatenate(
            [
                ops.ones((ops.shape(mkt_volumes)[0], 1, ops.shape(mkt_volumes)[2], 4, 1)),
                1. - ops.cast(keras.ops.cumsum(mkt_volumes, axis=1) > 0., dtype=y_true.dtype)[:,:-1]
            ], axis=1
        ), (1, 1, 1, 1, 2)
    )
    needed_execution_mask = (
        mask 
        + (1 - mask)
        * mkt_minimum_trade_mask
    )
    # Compute soft masks for all channel using cumulative sum 
    effective_execution_mask = (
        needed_execution_mask
        + (1 - needed_execution_mask)
        * ops.cast(ops.flip(keras.ops.cumsum(ops.flip(y_pred, axis=1), axis=1), axis=1) > 0., dtype=y_true.dtype)
    )
    eps = keras.backend.epsilon()

    # Volume Section:
    executed_volumes = y_pred[...,0:1] * effective_execution_mask[...,0:1]
    executed_notionals = executed_volumes * mkt_prices
    price_achieved_vto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    # Notional Section:
    executed_notionals = y_pred[...,1:2] * effective_execution_mask[...,1:2]
    executed_volumes = executed_notionals / mkt_prices
    price_achieved_nto = ops.sum(executed_notionals, axis=1) / (ops.sum(executed_volumes, axis=1) + eps)

    price_achieved = ops.concatenate([price_achieved_vto, price_achieved_nto], axis=3)
    
    mkt_achieved_volumes = mkt_volumes * effective_execution_mask
    mkt_achieved_notionals = mkt_achieved_volumes * mkt_prices
    mkt_benchmarks = ops.sum(mkt_achieved_notionals, axis=1) / (ops.sum(mkt_achieved_volumes, axis=1) + eps)

    diffs = (price_achieved / mkt_benchmarks - 1.) * 100.0
    risk_diff = (keras.ops.abs(diffs[:,:-1,1,:]) + keras.ops.abs(diffs[:,:-1,3,:]))/2.
    
    loss = keras.ops.mean(risk_diff)
    return loss * 100.