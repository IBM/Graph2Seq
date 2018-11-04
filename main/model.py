import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import tensorflow.contrib.seq2seq as seq2seq

from neigh_samplers import UniformNeighborSampler
from aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator
import numpy as np
import match_utils

class Graph2SeqNN(object):

    PAD = 0
    GO = 1
    EOS = 2

    def __init__(self, mode, conf, path_embed_method):

        self.mode = mode
        self.word_vocab_size = conf.word_vocab_size
        self.l2_lambda = conf.l2_lambda
        self.path_embed_method = path_embed_method
        # self.word_embedding_dim = conf.word_embedding_dim
        self.word_embedding_dim = conf.hidden_layer_dim
        self.encoder_hidden_dim = conf.encoder_hidden_dim

        # the setting for the GCN
        self.num_layers_decode = conf.num_layers_decode
        self.num_layers = conf.num_layers
        self.graph_encode_direction = conf.graph_encode_direction
        self.sample_layer_size = conf.sample_layer_size
        self.hidden_layer_dim = conf.hidden_layer_dim
        self.concat = conf.concat

        # the setting for the decoder
        self.beam_width = conf.beam_width
        self.decoder_type = conf.decoder_type
        self.seq_max_len = conf.seq_max_len

        self._text = tf.placeholder(tf.int32, [None, None])
        self.decoder_seq_length = tf.placeholder(tf.int32, [None])
        self.loss_weights = tf.placeholder(tf.float32, [None, None])

        # the following place holders are for the gcn
        self.fw_adj_info = tf.placeholder(tf.int32, [None, None])               # the fw adj info for each node
        self.bw_adj_info = tf.placeholder(tf.int32, [None, None])               # the bw adj info for each node
        self.feature_info = tf.placeholder(tf.int32, [None, None])              # the feature info for each node
        self.batch_nodes = tf.placeholder(tf.int32, [None, None])               # the nodes for each batch

        self.sample_size_per_layer = tf.shape(self.fw_adj_info)[1]

        self.single_graph_nodes_size = tf.shape(self.batch_nodes)[1]
        self.attention = conf.attention
        self.dropout = conf.dropout
        self.fw_aggregators = []
        self.bw_aggregators = []

        self.if_pred_on_dev = False

        self.learning_rate = conf.learning_rate

    def _init_decoder_train_connectors(self):
        batch_size, sequence_size = tf.unstack(tf.shape(self._text))
        self.batch_size = batch_size
        GO_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.GO
        EOS_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.PAD
        self.decoder_train_inputs = tf.concat([GO_SLICE, self._text], axis=1)
        self.decoder_train_length = self.decoder_seq_length + 1
        decoder_train_targets = tf.concat([self._text, EOS_SLICE], axis=1)
        _, decoder_train_targets_seq_len = tf.unstack(tf.shape(decoder_train_targets))
        decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1, decoder_train_targets_seq_len,
                                                    on_value=self.EOS, off_value=self.PAD, dtype=tf.int32)
        self.decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
        self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.decoder_train_inputs)



    def encode(self):
        with tf.variable_scope("embedding_layer"):
            pad_word_embedding = tf.zeros([1, self.word_embedding_dim])  # this is for the PAD symbol
            self.word_embeddings = tf.concat([pad_word_embedding,
                                              tf.get_variable('W_train', shape=[self.word_vocab_size,self.word_embedding_dim],
                                                                            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)], 0)

        with tf.variable_scope("graph_encoding_layer"):

            # self.encoder_outputs, self.encoder_state = self.gcn_encode()

            # this is for optimizing gcn
            encoder_outputs, encoder_state = self.optimized_gcn_encode()

            source_sequence_length = tf.reshape(
                tf.ones([tf.shape(encoder_outputs)[0], 1], dtype=tf.int32) * self.single_graph_nodes_size,
                (tf.shape(encoder_outputs)[0],))

            return encoder_outputs, encoder_state, source_sequence_length

    def encode_node_feature(self, word_embeddings, feature_info):
        # in some cases, we can use LSTM to produce the node feature representation
        # cell = self._build_encoder_cell(conf.num_layers, conf.dim)


        feature_embedded_chars = tf.nn.embedding_lookup(word_embeddings, feature_info)
        batch_size = tf.shape(feature_embedded_chars)[0]

        # node_repres = match_utils.multi_highway_layer(feature_embedded_chars, self.hidden_layer_dim, num_layers=1)
        # node_repres = tf.reshape(node_repres, [batch_size, -1])
        # y = tf.shape(node_repres)[1]
        #
        # node_repres = tf.concat([tf.slice(node_repres, [0,0], [batch_size-1, y]), tf.zeros([1, y])], 0)

        node_repres = tf.reshape(feature_embedded_chars, [batch_size, -1])

        return node_repres

    def optimized_gcn_encode(self):
        # [node_size, hidden_layer_dim]
        embedded_node_rep = self.encode_node_feature(self.word_embeddings, self.feature_info)

        fw_sampler = UniformNeighborSampler(self.fw_adj_info)
        bw_sampler = UniformNeighborSampler(self.bw_adj_info)
        nodes = tf.reshape(self.batch_nodes, [-1, ])

        # batch_size = tf.shape(nodes)[0]

        # the fw_hidden and bw_hidden is the initial node embedding
        # [node_size, dim_size]
        fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)

        # [node_size, adj_size]
        fw_sampled_neighbors = fw_sampler((nodes, self.sample_size_per_layer))
        bw_sampled_neighbors = bw_sampler((nodes, self.sample_size_per_layer))

        fw_sampled_neighbors_len = tf.constant(0)
        bw_sampled_neighbors_len = tf.constant(0)

        # sample
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 2

            if layer > 6:
                fw_aggregator = self.fw_aggregators[6]
            else:
                fw_aggregator = MeanAggregator(dim_mul * self.hidden_layer_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                self.fw_aggregators.append(fw_aggregator)

            # [node_size, adj_size, word_embedding_dim]
            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)

                # compute the neighbor size
                tmp_sum = tf.reduce_sum(tf.nn.relu(neigh_vec_hidden), axis=2)
                tmp_mask = tf.sign(tmp_sum)
                fw_sampled_neighbors_len = tf.reduce_sum(tmp_mask, axis=1)

            else:
                neigh_vec_hidden = tf.nn.embedding_lookup(
                    tf.concat([fw_hidden, tf.zeros([1, dim_mul * self.hidden_layer_dim])], 0), fw_sampled_neighbors)

            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))


            if self.graph_encode_direction == "bi":
                if layer > 6:
                    bw_aggregator = self.bw_aggregators[6]
                else:
                    bw_aggregator = MeanAggregator(dim_mul * self.hidden_layer_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                    self.bw_aggregators.append(bw_aggregator)

                if layer == 0:
                    neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)

                    # compute the neighbor size
                    tmp_sum = tf.reduce_sum(tf.nn.relu(neigh_vec_hidden), axis=2)
                    tmp_mask = tf.sign(tmp_sum)
                    bw_sampled_neighbors_len = tf.reduce_sum(tmp_mask, axis=1)

                else:
                    neigh_vec_hidden = tf.nn.embedding_lookup(
                        tf.concat([bw_hidden, tf.zeros([1, dim_mul * self.hidden_layer_dim])], 0), bw_sampled_neighbors)

                bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))

        # hidden stores the representation for all nodes
        fw_hidden = tf.reshape(fw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
        if self.graph_encode_direction == "bi":
            bw_hidden = tf.reshape(bw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
            hidden = tf.concat([fw_hidden, bw_hidden], axis=2)
        else:
            hidden = fw_hidden

        hidden = tf.nn.relu(hidden)

        pooled = tf.reduce_max(hidden, 1)
        if self.graph_encode_direction == "bi":
            graph_embedding = tf.reshape(pooled, [-1, 4 * self.hidden_layer_dim])
        else:
            graph_embedding = tf.reshape(pooled, [-1, 2 * self.hidden_layer_dim])

        graph_embedding = LSTMStateTuple(c=graph_embedding, h=graph_embedding)

        # shape of hidden: [batch_size, single_graph_nodes_size, 4 * hidden_layer_dim]
        # shape of graph_embedding: ([batch_size, 4 * hidden_layer_dim], [batch_size, 4 * hidden_layer_dim])
        return hidden, graph_embedding

    def decode(self, encoder_outputs, encoder_state, source_sequence_length):
        with tf.variable_scope("Decoder") as scope:
            beam_width = self.beam_width
            decoder_type = self.decoder_type
            seq_max_len = self.seq_max_len
            batch_size = tf.shape(encoder_outputs)[0]

            if self.path_embed_method == "lstm":
                self.decoder_cell = self._build_decode_cell()
                if self.mode == "test" and beam_width > 0:
                    memory = seq2seq.tile_batch(self.encoder_outputs, multiplier=beam_width)
                    source_sequence_length = seq2seq.tile_batch(self.source_sequence_length, multiplier=beam_width)
                    encoder_state = seq2seq.tile_batch(self.encoder_state, multiplier=beam_width)
                    batch_size = self.batch_size * beam_width
                else:
                    memory = encoder_outputs
                    source_sequence_length = source_sequence_length
                    encoder_state = encoder_state

                attention_mechanism = seq2seq.BahdanauAttention(self.hidden_layer_dim, memory,
                                                                memory_sequence_length=source_sequence_length)
                self.decoder_cell = seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism,
                                                             attention_layer_size=self.hidden_layer_dim)
                self.decoder_initial_state = self.decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            projection_layer = Dense(self.word_vocab_size, use_bias=False)

            """For training the model"""
            if self.mode == "train":
                decoder_train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_train_inputs_embedded,
                                                                         self.decoder_train_length)
                decoder_train = seq2seq.BasicDecoder(self.decoder_cell, decoder_train_helper,
                                                     self.decoder_initial_state,
                                                     projection_layer)
                decoder_outputs_train, decoder_states_train, decoder_seq_len_train = seq2seq.dynamic_decode(decoder_train)
                decoder_logits_train = decoder_outputs_train.rnn_output
                self.decoder_logits_train = tf.reshape(decoder_logits_train, [batch_size, -1, self.word_vocab_size])

            """For test the model"""
            # if self.mode == "infer" or self.if_pred_on_dev:
            if decoder_type == "greedy":
                decoder_infer_helper = seq2seq.GreedyEmbeddingHelper(self.word_embeddings,
                                                                     tf.ones([batch_size], dtype=tf.int32),
                                                                     self.EOS)
                decoder_infer = seq2seq.BasicDecoder(self.decoder_cell, decoder_infer_helper,
                                                     self.decoder_initial_state, projection_layer)
            elif decoder_type == "beam":
                decoder_infer = seq2seq.BeamSearchDecoder(cell=self.decoder_cell, embedding=self.word_embeddings,
                                                          start_tokens=tf.ones([batch_size], dtype=tf.int32),
                                                          end_token=self.EOS,
                                                          initial_state=self.decoder_initial_state,
                                                          beam_width=beam_width,
                                                          output_layer=projection_layer)

            decoder_outputs_infer, decoder_states_infer, decoder_seq_len_infer = seq2seq.dynamic_decode(decoder_infer,
                                                                                                        maximum_iterations=seq_max_len)

            if decoder_type == "beam":
                self.decoder_logits_infer = tf.no_op()
                self.sample_id = decoder_outputs_infer.predicted_ids

            elif decoder_type == "greedy":
                self.decoder_logits_infer = decoder_outputs_infer.rnn_output
                self.sample_id = decoder_outputs_infer.sample_id

    def _build_decode_cell(self):
        if self.num_layers == 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=4*self.hidden_layer_dim)
            if self.mode == "train":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1 - self.dropout)
            return cell
        else:
            cell_list = []
            for i in range(self.num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(self._decoder_hidden_size)
                if self.mode == "train":
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, 1 - self.dropout)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_encoder_cell(self, num_layers, hidden_layer_dim):
        if num_layers == 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_dim)
            if self.mode == "train":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1 - self.dropout)
            return cell
        else:
            cell_list = []
            for i in range(num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_dim)
                if self.mode == "train":
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, 1 - self.dropout)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _init_optimizer(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_train_targets, logits=self.decoder_logits_train)
        decode_loss = (tf.reduce_sum(crossent * self.loss_weights) / tf.cast(self.batch_size, tf.float32))

        train_loss = decode_loss

        for aggregator in self.fw_aggregators:
            for var in aggregator.vars.values():
                train_loss += self.l2_lambda * tf.nn.l2_loss(var)

        for aggregator in self.bw_aggregators:
            for var in aggregator.vars.values():
                train_loss += self.l2_lambda * tf.nn.l2_loss(var)

        self.loss_op = train_loss
        self.cross_entropy_sum = train_loss
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def _build_graph(self):
        encoder_outputs, encoder_state, source_sequence_length = self.encode()

        if self.mode == "train":
            self._init_decoder_train_connectors()

        self.decode(encoder_outputs=encoder_outputs, encoder_state=encoder_state, source_sequence_length=source_sequence_length)

        if self.mode == "train":
            self._init_optimizer()

    def act(self, sess, mode, dict, if_pred_on_dev):
        text = np.array(dict['seq'])
        decoder_seq_length = np.array(dict['decoder_seq_length'])
        loss_weights = np.array(dict['loss_weights'])
        batch_graph = dict['batch_graph']
        fw_adj_info = batch_graph['g_fw_adj']
        bw_adj_info = batch_graph['g_bw_adj']
        feature_info = batch_graph['g_ids_features']
        batch_nodes = batch_graph['g_nodes']

        self.if_pred_on_dev = if_pred_on_dev

        feed_dict = {
            self._text: text,
            self.decoder_seq_length: decoder_seq_length,
            self.loss_weights: loss_weights,
            self.fw_adj_info: fw_adj_info,
            self.bw_adj_info: bw_adj_info,
            self.feature_info: feature_info,
            self.batch_nodes: batch_nodes
        }

        if mode == "train" and not if_pred_on_dev:
            output_feeds = [self.train_op, self.loss_op, self.cross_entropy_sum]
        elif mode == "test" or if_pred_on_dev:
            output_feeds = [self.sample_id]

        results = sess.run(output_feeds, feed_dict)
        return results