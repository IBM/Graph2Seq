import tensorflow as tf
from layers import Layer, Dense
from inits import glorot, zeros
from pooling import mean_pool

class GatedMeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, **kwargs):
        super(GatedMeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

            self.vars['gate_weights'] = glorot([2*output_dim, 2*output_dim],
                                                name='gate_weights')
            self.vars['gate_bias'] = zeros([2*output_dim], name='bias')


        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        gate = tf.concat([from_self, from_neighs], axis=1)
        gate = tf.matmul(gate, self.vars["gate_weights"]) + self.vars["gate_bias"]
        gate = tf.nn.relu(gate)

        return gate*self.act(output)

class MeanAggregator(Layer):
    """Aggregates via mean followed by matmul and non-linearity."""

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train", **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, neigh_len = inputs

        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # reduce_mean performs better than mean_pool
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        # neigh_means = mean_pool(neigh_vecs, neigh_len)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions."""
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        if model_size == "small":
            hidden_dim = self.hidden_dim = 50
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 50

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim, output_dim=hidden_dim, act=tf.nn.relu,
                                     dropout=dropout, sparse_inputs=False, logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):

            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim], name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
        return self.act(output)