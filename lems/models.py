import keras
from keras import ops
from keras.initializers import Initializer
from keras import Model, Sequential
from keras.constraints import Constraint
from keras.layers import (
    Layer,
    Add,
    LayerNormalization,
    Dense,
    Multiply,
    Reshape,
    Activation,
    MultiHeadAttention
)
from tkan import TKAN

@keras.utils.register_keras_serializable(name="EqualInitializer")
class EqualInitializer(Initializer):
    """Initializes weights to 1/n_ahead."""
    
    def __init__(self, n_ahead):
        self.n_ahead = n_ahead
        
    def __call__(self, shape, dtype=None):
        return ops.ones(shape, dtype=dtype) / self.n_ahead

        
    def get_config(self):
        return {'n_ahead': self.n_ahead}


@keras.utils.register_keras_serializable(name="PositiveSumToOneConstraint")
class PositiveSumToOneConstraint(keras.constraints.Constraint):
    """Constrains the weights to be positive and sum to 1."""
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def __call__(self, w):
        # First ensure values are positive
        w = keras.activations.relu(w)
        # Then normalize to sum to 1
        return w / (keras.ops.sum(w, axis=self.axis, keepdims=True) + keras.backend.epsilon())

    def get_config(self):
        return {}

@keras.utils.register_keras_serializable(name="AddAndNorm")
class AddAndNorm(Layer):
    def __init__(self, **kwargs):
        super(AddAndNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.add_layer = Add()
        self.add_layer.build(input_shape)
        self.norm_layer = LayerNormalization()
        self.norm_layer.build(self.add_layer.compute_output_shape(input_shape))
    
    def call(self, inputs):
        tmp = self.add_layer(inputs)
        tmp = self.norm_layer(tmp)
        return tmp

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Assuming all input shapes are the same

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable(name="GRN")
class Gate(Layer):
    def __init__(self, hidden_layer_size = None, **kwargs):
        super(Gate, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        

    def build(self, input_shape):
        if self.hidden_layer_size is None:
            self.hidden_layer_size = input_shape[-1]
        self.dense_layer = Dense(self.hidden_layer_size)
        self.gated_layer = Dense(self.hidden_layer_size, activation='sigmoid')
        self.dense_layer.build(input_shape)
        self.gated_layer.build(input_shape)
        self.multiply = Multiply()

    def call(self, inputs):
        dense_output = self.dense_layer(inputs)
        gated_output = self.gated_layer(inputs)
        return ops.multiply(dense_output, gated_output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_layer_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_layer_size': self.hidden_layer_size,
        })
        return config


@keras.utils.register_keras_serializable(name="GRN")
class GRN(Layer):
    def __init__(self, hidden_layer_size, output_size=None, **kwargs):
        super(GRN, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

    def build(self, input_shape):
        if self.output_size is None:
            self.output_size = self.hidden_layer_size
        self.skip_layer = Dense(self.output_size)
        self.skip_layer.build(input_shape)
        
        self.hidden_layer_1 = Dense(self.hidden_layer_size, activation='elu')
        self.hidden_layer_1.build(input_shape)
        self.hidden_layer_2 = Dense(self.hidden_layer_size)
        self.hidden_layer_2.build((*input_shape[:2], self.hidden_layer_size))
        self.gate_layer = Gate(self.output_size)
        self.gate_layer.build((*input_shape[:2], self.hidden_layer_size))
        self.add_and_norm_layer = AddAndNorm()
        self.add_and_norm_layer.build([(*input_shape[:2], self.output_size),(*input_shape[:2], self.output_size)])

    def call(self, inputs):
        skip = self.skip_layer(inputs)
        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        gating_output = self.gate_layer(hidden)
        return self.add_and_norm_layer([skip, gating_output])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_layer_size': self.hidden_layer_size,
            'output_size': self.output_size,
        })
        return config


@keras.utils.register_keras_serializable(name="VariableSelectionNetwork")
class VariableSelectionNetwork(Layer):
    def __init__(self, num_hidden, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        batch_size, time_steps, embedding_dim, num_inputs = input_shape
        self.softmax = Activation('softmax')
        self.num_inputs = num_inputs
        self.flatten_dim = time_steps * embedding_dim * num_inputs
        self.reshape_layer = Reshape(target_shape=[time_steps, embedding_dim * num_inputs])
        self.reshape_layer.build(input_shape)
        self.mlp_dense = GRN(hidden_layer_size = self.num_hidden, output_size=num_inputs)
        self.mlp_dense.build((batch_size, time_steps, embedding_dim * num_inputs))
        self.grn_layers = [GRN(self.num_hidden) for _ in range(num_inputs)]
        for i in range(num_inputs):
            self.grn_layers[i].build(input_shape[:3])
        super(VariableSelectionNetwork, self).build(input_shape)

    def call(self, inputs):
        _, time_steps, embedding_dim, num_inputs = inputs.shape
        flatten = self.reshape_layer(inputs)
        # Variable selection weights
        mlp_outputs = self.mlp_dense(flatten)
        sparse_weights = keras.activations.softmax(mlp_outputs)
        sparse_weights = ops.expand_dims(sparse_weights, axis=2)
        
        # Non-linear Processing & weight application
        trans_emb_list = []
        for i in range(num_inputs):
            grn_output = self.grn_layers[i](inputs[:, :, :, i])
            trans_emb_list.append(grn_output)
        
        transformed_embedding = ops.stack(trans_emb_list, axis=-1)
        combined = ops.multiply(sparse_weights, transformed_embedding)
        temporal_ctx = ops.sum(combined, axis=-1)
        
        return temporal_ctx

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_hidden': self.num_hidden,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_hidden)


@keras.utils.register_keras_serializable(name="EmbeddingLayer")
class EmbeddingLayer(Layer):
    def __init__(self, num_hidden, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        self.dense_layers = [
            Dense(self.num_hidden) for _ in range(input_shape[-1])
        ]
        for i in range(input_shape[-1]):
            self.dense_layers[i].build((*input_shape[:2], 1))
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        embeddings = [dense_layer(inputs[:, :, i:i+1]) for i, dense_layer in enumerate(self.dense_layers)]
        return ops.stack(embeddings, axis=-1)

    def compute_output_shape(self, input_shape):
        return list(input_shape[:-1]) + [self.num_hidden, input_shape[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_hidden': self.num_hidden,
        })
        return config

@keras.utils.register_keras_serializable(name="FusedInitializer")
class FusedInitializer(keras.initializers.Initializer):
    def __init__(self, base_initializer):
        # Get the base initializer (e.g., "glorot_uniform")
        self.base_initializer = keras.initializers.get(base_initializer)

    def __call__(self, shape, dtype=None, **kwargs):
        # shape is expected to be (n_fused, features, units)
        # We want to initialize as if the shape were (features, units)
        base_shape = shape[2:]
        # Generate the kernel for one branch
        single_kernel = self.base_initializer(base_shape, dtype=dtype, **kwargs)
        # Replicate along the fused dimension
        return ops.tile(ops.expand_dims(single_kernel, axis=(0, 1)), [shape[0],shape[1], 1, 1])

    def get_config(self):
        return {'base_initializer': keras.initializers.serialize(self.base_initializer)}

@keras.utils.register_keras_serializable(name="FusedMLP")
class FusedMLP(Model):
    def __init__(self, n_models, n_fused, config, *args, **kwargs):
        """
        n_models: int - different models based on the model column selected - typically here min period
        n_fused: int - number of parrallele different models - here buy volume, sell volume, buy notional....
        config: Iterable of tuple(units: int, activation: str or keras activation, initializer: str or keras initializer
        """
        super().__init__(*args, **kwargs)
        self.n_models = n_models
        self.n_fused = n_fused
        self.config = config
        
        
    def build(self, input_shape):
        _, n_model, features = input_shape
        assert n_model == self.n_models
        self.kernels = []
        self.activations = []
        self.bias = []
        for idx, (units, activation, initializer) in enumerate(self.config):
            self.kernels.append(self.add_weight(
                shape=(self.n_models, self.n_fused, features, units),
                name=f"kernel_{idx}",
                initializer=FusedInitializer(keras.initializers.get(initializer)),
                trainable=True,
            ))
            self.bias.append(self.add_weight(
                shape=(self.n_models, self.n_fused, units),
                name=f"bias_{idx}",
                initializer="zeros",
                trainable=True,
            ))
            self.activations.append(keras.activations.get(activation))
            features = units
        super().build(input_shape)            

    def call(self, inputs):
        x = ops.tile(inputs[:,:,None,:], [1, 1, self.n_fused, 1])
        for kernel, bias, activation in zip(self.kernels, self.bias, self.activations):
            x = activation(ops.einsum("bnfi,nfij->bnfj", x, kernel) + bias)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_fused, self.config[-1][0])

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_models': self.n_models,
            'n_fused': self.n_fused,
            'config': self.config,
        })
        return config

@keras.utils.register_keras_serializable(name="ClipConstraint")
class ClipConstraint(keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return keras.ops.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}



@keras.utils.register_keras_serializable(name="LargeExecutionModel")
class LargeExecutionModel(Model):
    def __init__(self, lookback, n_ahead, hidden_size=200, hidden_rnn_layer=2, num_heads=8, num_embedding=20, max_scale_factor=1.5, fused_mlp_hidden_dim = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookback = lookback
        self.n_ahead = n_ahead
        self.max_scale_factor = max_scale_factor
        self.fused_mlp_hidden_dim = fused_mlp_hidden_dim or hidden_size
        self.hidden_size = hidden_size
        self.hidden_rnn_layer = hidden_rnn_layer
        self.num_embedding = num_embedding
        self.num_heads = num_heads
        
    def build(self, input_shape):
        base_input_shape, minimum_trading_rates_shape, maximum_trading_rates_shape = input_shape
        assert minimum_trading_rates_shape == maximum_trading_rates_shape 
        assert minimum_trading_rates_shape[0] == base_input_shape[0]
        standard_input_shape = (base_input_shape[0], base_input_shape[1], base_input_shape[2])
        feature_shape = standard_input_shape
        assert feature_shape[1] == self.lookback + self.n_ahead - 1
        
        self.embedding = EmbeddingLayer(self.num_embedding)
        self.embedding.build(standard_input_shape)
        embedding_output_shape = self.embedding.compute_output_shape(standard_input_shape)
        self.vsn = VariableSelectionNetwork(self.hidden_size)
        self.vsn.build(embedding_output_shape)
        vsn_output_shape = (standard_input_shape[0], standard_input_shape[1], self.hidden_size)
        
        # RNN layers
        self.internal_rnn = Sequential([
            TKAN(self.hidden_size, return_sequences=True)
            for _ in range(self.hidden_rnn_layer)
        ])
        self.internal_rnn.build(vsn_output_shape)
        self.gate = Gate()
        self.gate.build(vsn_output_shape)
        self.addnorm = AddAndNorm()
        self.addnorm.build([vsn_output_shape,vsn_output_shape])
        self.grn = GRN(self.hidden_size)
        self.grn.build(vsn_output_shape)
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads,
            value_dim=self.hidden_size // self.num_heads,
            use_bias=True
        )
        self.attention.build(vsn_output_shape, vsn_output_shape, vsn_output_shape)#as the tkan do not changes shapes
        
        # Dense layers for volume prediction
        self.internal_hidden_to_raw_allocation = tuple(
                FusedMLP(
                    n_models = self.n_ahead + 1, # different internal model for each minimum end period possible
                    n_fused = 8, # One model for each sub case (volume ou notionel, vwap ou twap,  buy ou sell..)
                    config=(
                        (self.fused_mlp_hidden_dim, 'relu', 'glorot_uniform'),
                        (self.fused_mlp_hidden_dim, 'relu', 'glorot_uniform'),
                        (1, 'softsign', 'lecun_normal'),
                    )
                )
            for _ in range(self.n_ahead - 1)
        )
           
        self.scale_factor = self.add_weight(
            shape=(1,self.n_ahead + 1, 1, 1),
            name="buy_factor",
            initializer="ones",
            trainable=True,
            constraint=ClipConstraint(0.5, self.max_scale_factor)
        )

        for step in range(self.n_ahead - 1):
            self.internal_hidden_to_raw_allocation[step].build((feature_shape[0], self.n_ahead + 1, self.hidden_size + standard_input_shape[2] + step * 18))
            
        # Base volume curve
        self.base_volume_curve = self.add_weight(
            shape=(self.n_ahead, self.n_ahead + 1,2), # Per minimum period, step, VWAP or TWAP
            name="base_volume_curve",
            initializer=EqualInitializer(self.n_ahead),
            constraint=PositiveSumToOneConstraint(axis=1),
            trainable=True
        )
        self.base_notional_curve = self.add_weight(
            shape=(self.n_ahead, self.n_ahead + 1, 2), # Per minimum period, step, VWAP or TWAP
            name="base_notional_curve",
            initializer=EqualInitializer(self.n_ahead),
            constraint=PositiveSumToOneConstraint(axis=1),
            trainable=True
        )
        super(LargeExecutionModel, self).build(input_shape)

    @staticmethod
    def soft_clip(x, upper, scale=10.0):
        # When x << upper, sigmoid(scale*(upper - x)) ~ 1 => soft_clip(x) ~ x.
        # When x >> upper, sigmoid(scale*(upper - x)) ~ 0 => soft_clip(x) ~ upper.
        ratio = keras.activations.sigmoid(scale * (upper - x))
        return x * ratio + upper * (1 - ratio)
    
    def call(self, inputs):
        # Last features of inputs are price, volumes, minimum trading rate, maximum trading rate, minimum end period - very important here
        inputs, minimum_trading_rates, maximum_trading_rates = inputs
        batch_size = ops.shape(inputs)[0]
        prices = inputs[...,-2:-1]
        volumes = inputs[...,-1:]
        standard_inputs = ops.concatenate([
            inputs[...,:-2],
            ops.tile(minimum_trading_rates[:,None,:], (1, ops.shape(inputs)[1], 1)),
            ops.tile(maximum_trading_rates[:,None,:], (1, ops.shape(inputs)[1], 1)),
        ], axis=2)
        
        embedded_features = self.embedding(standard_inputs)
        selected = self.vsn(embedded_features)
        # Get RNN hidden states
        rnn_hidden = self.internal_rnn(selected)
        all_context = self.addnorm([self.gate(rnn_hidden), selected])
        enriched = self.grn(all_context)
        
        # Apply causal self-attention
        attended_hidden = self.attention(
            query=enriched,
            value=enriched,
            key=enriched,
            use_causal_mask=True
        )

        total_vto_volume, total_vto_notional, total_nto_volume, total_nto_notional = ops.zeros((batch_size, self.n_ahead + 1, 4, 1)), ops.zeros((batch_size, self.n_ahead + 1, 4, 1)), ops.zeros((batch_size, self.n_ahead + 1, 4, 1)), ops.zeros((batch_size, self.n_ahead + 1, 4, 1))
        total_mkt_volume, total_mkt_notional, total_mkt_prices = ops.zeros((batch_size, 1)), ops.zeros((batch_size, 1)), ops.zeros((batch_size, 1))

        vto_allocation_curve, vto_price_curve, nto_allocation_curve, nto_price_curve = ops.zeros((batch_size, self.n_ahead + 1, 4, 0)), ops.zeros((batch_size, self.n_ahead + 1, 4, 0)), ops.zeros((batch_size, self.n_ahead + 1, 4, 0)), ops.zeros((batch_size, self.n_ahead + 1, 4, 0))
        mkt_vwap_curve, mkt_twap_curve = ops.zeros((batch_size, 0)), ops.zeros((batch_size, 0))
        
        minimum_trading_rates, maximum_trading_rates = ops.expand_dims(minimum_trading_rates, axis=(1,2)), ops.expand_dims(maximum_trading_rates, axis=(1,2))
        for step in range(0, self.n_ahead - 2):
            # First Prepare Market parts - Not used in prediction of this step but to prepare inputs of next step! 
            # I did it here in order to have mkt_price and mkt_volume to use in the Volume Allocation and Notional Allocation section directly
            mkt_price = prices[:, self.lookback + step + 1]
            mkt_volume = volumes[:, self.lookback + step + 1]
            total_mkt_prices = total_mkt_prices + mkt_price
            total_mkt_volume = total_mkt_volume + mkt_volume
            total_mkt_notional = total_mkt_notional + mkt_volume * mkt_price
            current_mkt_vwap = total_mkt_notional / (total_mkt_volume + keras.backend.epsilon()) 
            current_mkt_twap = total_mkt_prices / (step + 1)

            # Part for Volume Allocation (BUY vs VWAP, SELL vs VWAP, BUY vs TWAP, SELL vs TWAP)
            factors = ops.power(
                        (1. + self.internal_hidden_to_raw_allocation[step](
                                ops.concatenate([
                                    ops.tile(attended_hidden[:, self.lookback + step:self.lookback + step + 1, :], (1, self.n_ahead + 1, 1)),
                                    ops.tile(standard_inputs[:, self.lookback + step:self.lookback + step + 1, :], (1, self.n_ahead + 1, 1)),
                                    ops.reshape(vto_allocation_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                    ops.reshape(nto_allocation_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                    ops.reshape(vto_price_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                    ops.reshape(nto_price_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                    ops.tile(mkt_vwap_curve[:,None,:], (1, self.n_ahead + 1, 1)),
                                    ops.tile(mkt_twap_curve[:,None,:], (1, self.n_ahead + 1, 1)),
                                ], axis=2)
                            )
                        ),
                        self.scale_factor
                    )
            volume_factors, notionals_factors = ops.split(factors, 2, axis=2)
            raw_estimated_volume = volume_factors * ops.tile(self.base_volume_curve[step], (1, 1,2))[:,:,:, None]
            raw_estimated_notionals = notionals_factors * ops.tile(self.base_notional_curve[step], (1, 1,2))[:,:,:, None]
            remainings = 1. - total_vto_volume
            smoothed_for_max_trading = self.soft_clip(raw_estimated_volume + minimum_trading_rates, maximum_trading_rates, scale=10.0)
            minimum_to_trades = ops.maximum(remainings / (self.n_ahead - step) - maximum_trading_rates, 0.)
            smoothed = self.soft_clip(smoothed_for_max_trading + minimum_to_trades, remainings, scale=10.0)
            estimated = keras.ops.clip(smoothed, 0., remainings)
            total_vto_volume = total_vto_volume + estimated
            total_vto_notional = total_vto_notional + estimated * mkt_price[...,None,None]

            current_vto_price = total_vto_notional / (total_vto_volume + keras.backend.epsilon())

            vto_allocation_curve = ops.concatenate([vto_allocation_curve, estimated], axis=3)
            vto_price_curve = ops.concatenate([vto_price_curve, current_vto_price], axis=3)
            
            # Part for Notional Allocation (BUY vs VWAP, SELL vs VWAP, BUY vs TWAP, SELL vs TWAP)
            remainings = 1. - total_nto_notional
            smoothed_for_max_trading = self.soft_clip(raw_estimated_notionals + minimum_trading_rates, maximum_trading_rates, scale=10.0)
            minimum_to_trades = ops.maximum(remainings / (self.n_ahead - step) - maximum_trading_rates, 0.)
            smoothed = self.soft_clip(smoothed_for_max_trading + minimum_to_trades, remainings, scale=10.0)
            estimated = keras.ops.clip(smoothed, 0., remainings)
            total_nto_notional = total_nto_notional + estimated
            total_nto_volume = total_nto_volume + estimated / mkt_price[...,None,None]
            
            current_nto_price = total_nto_notional / (total_nto_volume + keras.backend.epsilon())
            nto_allocation_curve = ops.concatenate([nto_allocation_curve, estimated], axis=3)
            nto_price_curve = ops.concatenate([nto_price_curve, total_nto_notional], axis=3)

            mkt_vwap_curve = ops.concatenate([mkt_vwap_curve, current_mkt_vwap], axis=1)
            mkt_twap_curve = ops.concatenate([mkt_twap_curve, current_mkt_twap], axis=1)

        # Do the before last step outside of the loop to not have the need of using future prices to increment market price curve - market volume curves
        step = self.n_ahead - 2
        factors = ops.power(
                    (1. + self.internal_hidden_to_raw_allocation[step](
                            ops.concatenate([
                                ops.tile(attended_hidden[:, self.lookback + step:self.lookback + step + 1, :], (1, self.n_ahead + 1, 1)),
                                ops.tile(standard_inputs[:, self.lookback + step:self.lookback + step + 1, :], (1, self.n_ahead + 1, 1)),
                                ops.reshape(vto_allocation_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                ops.reshape(nto_allocation_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                ops.reshape(vto_price_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                ops.reshape(nto_price_curve, (batch_size, self.n_ahead + 1, 4 * step)),
                                ops.tile(mkt_vwap_curve[:,None,:], (1, self.n_ahead + 1, 1)),
                                ops.tile(mkt_twap_curve[:,None,:], (1, self.n_ahead + 1, 1)),
                            ], axis=2)
                        )
                    ),
                    self.scale_factor
                )
        volume_factors, notionals_factors = ops.split(factors, 2, axis=2)
        raw_estimated_volume = volume_factors * ops.tile(self.base_volume_curve[step], (1, 1,2))[:,:,:, None]
        raw_estimated_notionals = notionals_factors * ops.tile(self.base_notional_curve[step], (1, 1,2))[:,:,:, None]
        remainings = 1. - total_vto_volume
        smoothed_for_max_trading = self.soft_clip(raw_estimated_volume + minimum_trading_rates, maximum_trading_rates, scale=10.0)
        minimum_to_trades = ops.maximum(remainings / (self.n_ahead - step) - maximum_trading_rates, 0.)
        smoothed = self.soft_clip(smoothed_for_max_trading + minimum_to_trades, remainings, scale=10.0)
        estimated = keras.ops.clip(smoothed, 0., remainings)
        total_vto_volume = total_vto_volume + estimated
        vto_allocation_curve = ops.concatenate([vto_allocation_curve, estimated], axis=3)
        
        # Part for Notional Allocation (BUY vs VWAP, SELL vs VWAP, BUY vs TWAP, SELL vs TWAP)
        remainings = 1. - total_nto_notional
        smoothed_for_max_trading = self.soft_clip(raw_estimated_notionals + minimum_trading_rates, maximum_trading_rates, scale=10.0)
        minimum_to_trades = ops.maximum(remainings / (self.n_ahead - step) - maximum_trading_rates, 0.)
        smoothed = self.soft_clip(smoothed_for_max_trading + minimum_to_trades, remainings, scale=10.0)
        estimated = keras.ops.clip(smoothed, 0., remainings)
        total_nto_notional = total_nto_notional + estimated
        
        nto_allocation_curve = ops.concatenate([nto_allocation_curve, estimated], axis=3)
        
        vto_allocation_curve = ops.concatenate([vto_allocation_curve, 1. - total_vto_volume], axis=3)
        nto_allocation_curve = ops.concatenate([nto_allocation_curve, 1. - total_nto_notional], axis=3)

        # Output is shape (batch, n_ahead, n_ahead, 4, 2) - first n_ahead = steps, second n_ahead = minimum_period, 4 = BUY VWAP, BUY TWAP, SELL VWAP, SELL TWAP, 2=Volume, Notional
        return ops.moveaxis(ops.stack([vto_allocation_curve, nto_allocation_curve], axis=-1), 3, 1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lookback': self.lookback,
            'n_ahead': self.n_ahead,
            'hidden_size': self.hidden_size,
            'hidden_rnn_layer': self.hidden_rnn_layer,
            'max_scale_factor': self.max_scale_factor,
            'fused_mlp_hidden_dim': self.fused_mlp_hidden_dim,
            'num_embedding': self.num_embedding,
            'num_heads': self.num_heads
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2, self.n_ahead)

