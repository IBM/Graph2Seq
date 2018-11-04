import configure as conf
import data_collector as data_collector
import loaderAndwriter as disk_helper
import numpy as np
from model import Graph2SeqNN
import tensorflow as tf
import helpers as helpers
import datetime
import text_decoder
from evaluator import evaluate
import os
import argparse
import json

def main(mode):

    word_idx = {}

    if mode == "train":
        epochs = conf.epochs
        train_batch_size = conf.train_batch_size

        # read the training data from a file
        print("reading training data into the mem ...")
        texts_train, graphs_train = data_collector.read_data(conf.train_data_path, word_idx, if_increase_dict=True)

        print("reading development data into the mem ...")
        texts_dev, graphs_dev = data_collector.read_data(conf.dev_data_path, word_idx, if_increase_dict=False)

        print("writing word-idx mapping ...")
        disk_helper.write_word_idx(word_idx, conf.word_idx_file_path)

        print("vectoring training data ...")
        tv_train = data_collector.vectorize_data(word_idx, texts_train)

        print("vectoring dev data ...")
        tv_dev = data_collector.vectorize_data(word_idx, texts_dev)

        conf.word_vocab_size = len(word_idx.keys()) + 1

        with tf.Graph().as_default():
            with tf.Session() as sess:
                model = Graph2SeqNN("train", conf, path_embed_method="lstm")

                model._build_graph()
                saver = tf.train.Saver(max_to_keep=None)
                sess.run(tf.initialize_all_variables())

                def train_step(seqs, decoder_seq_length, loss_weights, batch_graph, if_pred_on_dev=False):
                    dict = {}
                    dict['seq'] = seqs
                    dict['batch_graph'] = batch_graph
                    dict['loss_weights'] = loss_weights
                    dict['decoder_seq_length'] = decoder_seq_length

                    if not if_pred_on_dev:
                        _, loss_op, cross_entropy = model.act(sess, "train", dict, if_pred_on_dev)
                        return loss_op, cross_entropy
                    else:
                        sample_id = model.act(sess, "train", dict, if_pred_on_dev)
                        return sample_id

                best_acc_on_dev = 0.0
                for t in range(1, epochs + 1):
                    n_train = len(texts_train)
                    temp_order = list(range(n_train))
                    np.random.shuffle(temp_order)

                    loss_sum = 0.0
                    for start in range(0, n_train, train_batch_size):
                        end = min(start+train_batch_size, n_train)
                        tv = []
                        graphs = []
                        for _ in range(start, end):
                            idx = temp_order[_]
                            tv.append(tv_train[idx])
                            graphs.append(graphs_train[idx])

                        batch_graph = data_collector.cons_batch_graph(graphs)
                        gv = data_collector.vectorize_batch_graph(batch_graph, word_idx)

                        tv, tv_real_len, loss_weights = helpers.batch(tv)

                        loss_op, cross_entropy = train_step(tv, tv_real_len, loss_weights, gv)
                        loss_sum += loss_op

                    #################### test the model on the dev data #########################
                    n_dev = len(texts_dev)
                    dev_batch_size = conf.dev_batch_size

                    idx_word = {}
                    for w in word_idx:
                        idx_word[word_idx[w]] = w

                    pred_texts = []
                    for start in range(0, n_dev, dev_batch_size):
                        end = min(start+dev_batch_size, n_dev)
                        tv = []
                        graphs = []
                        for _ in range(start, end):
                            tv.append(tv_dev[_])
                            graphs.append(graphs_dev[_])

                        batch_graph = data_collector.cons_batch_graph(graphs)
                        gv = data_collector.vectorize_batch_graph(batch_graph, word_idx)

                        tv, tv_real_len, loss_weights = helpers.batch(tv)

                        sample_id = train_step(tv, tv_real_len, loss_weights, gv, if_pred_on_dev=True)[0]

                        for tmp_id in sample_id:
                            pred_texts.append(text_decoder.decode_text(tmp_id, idx_word))

                    acc = evaluate(type="acc", golds=texts_dev, preds=pred_texts)
                    if_save_model = False
                    if acc >= best_acc_on_dev:
                        best_acc_on_dev = acc
                        if_save_model = True

                    time_str = datetime.datetime.now().isoformat()
                    print('-----------------------')
                    print('time:{}'.format(time_str))
                    print('Epoch', t)
                    print('Acc on Dev: {}'.format(acc))
                    print('Best acc on Dev: {}'.format(best_acc_on_dev))
                    print('Loss on train:{}'.format(loss_sum))
                    if if_save_model:
                        save_path = "../saved_model/"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        path = saver.save(sess, save_path + 'model', global_step=0)
                        print("Already saved model to {}".format(path))

                    print('-----------------------')

    elif mode == "test":
        test_batch_size = conf.test_batch_size

        # read the test data from a file
        print("reading test data into the mem ...")
        texts_test, graphs_test = data_collector.read_data(conf.test_data_path, word_idx, if_increase_dict=False)

        print("reading word idx mapping from file")
        word_idx = disk_helper.read_word_idx_from_file(conf.word_idx_file_path)

        idx_word = {}
        for w in word_idx:
            idx_word[word_idx[w]] = w

        print("vectoring test data ...")
        tv_test = data_collector.vectorize_data(word_idx, texts_test)

        conf.word_vocab_size = len(word_idx.keys()) + 1

        with tf.Graph().as_default():
            with tf.Session() as sess:
                model = Graph2SeqNN("test", conf, path_embed_method="lstm")
                model._build_graph()
                saver = tf.train.Saver(max_to_keep=None)

                model_path_name = "../saved_model/model-0"
                model_pred_path = "../saved_model/prediction.txt"

                saver.restore(sess, model_path_name)

                def test_step(seqs, decoder_seq_length, loss_weights, batch_graph):
                    dict = {}
                    dict['seq'] = seqs
                    dict['batch_graph'] = batch_graph
                    dict['loss_weights'] = loss_weights
                    dict['decoder_seq_length'] = decoder_seq_length
                    sample_id = model.act(sess, "test", dict, if_pred_on_dev=False)
                    return sample_id

                n_test = len(texts_test)

                pred_texts = []
                global_graphs = []
                for start in range(0, n_test, test_batch_size):
                    end = min(start + test_batch_size, n_test)
                    tv = []
                    graphs = []
                    for _ in range(start, end):
                        tv.append(tv_test[_])
                        graphs.append(graphs_test[_])
                        global_graphs.append(graphs_test[_])

                    batch_graph = data_collector.cons_batch_graph(graphs)
                    gv = data_collector.vectorize_batch_graph(batch_graph, word_idx)
                    tv, tv_real_len, loss_weights = helpers.batch(tv)

                    sample_id = test_step(tv, tv_real_len, loss_weights, gv)[0]
                    for tem_id in sample_id:
                        pred_texts.append(text_decoder.decode_text(tem_id, idx_word))

                acc = evaluate(type="acc", golds=texts_test, preds=pred_texts)
                print("acc on test set is {}".format(acc))

                # write prediction result into a file
                with open(model_pred_path, 'w+') as f:
                    for _ in range(len(global_graphs)):
                        f.write("graph:\t"+json.dumps(global_graphs[_])+"\nGold:\t"+texts_test[_]+"\nPredicted:\t"+pred_texts[_]+"\n")
                        if texts_test[_].strip() ==  pred_texts[_].strip():
                            f.write("Correct\n\n")
                        else:
                            f.write("Incorrect\n\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", type=str, choices=["train", "test"])
    argparser.add_argument("-sample_size_per_layer", type=int, default=4, help="sample size at each layer")
    argparser.add_argument("-sample_layer_size", type=int, default=4, help="sample layer size")
    argparser.add_argument("-epochs", type=int, default=100, help="training epochs")
    argparser.add_argument("-learning_rate", type=float, default=conf.learning_rate, help="learning rate")
    argparser.add_argument("-word_embedding_dim", type=int, default=conf.word_embedding_dim, help="word embedding dim")
    argparser.add_argument("-hidden_layer_dim", type=int, default=conf.hidden_layer_dim)

    config = argparser.parse_args()

    mode = config.mode
    conf.sample_layer_size = config.sample_layer_size
    conf.sample_size_per_layer = config.sample_size_per_layer
    conf.epochs = config.epochs
    conf.learning_rate = config.learning_rate
    conf.word_embedding_dim = config.word_embedding_dim
    conf.hidden_layer_dim = config.hidden_layer_dim

    main(mode)
