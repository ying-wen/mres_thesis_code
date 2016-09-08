from cat2vec import Options, Cat2Vec
import tensorflow as tf
from sample_encoding import *
from utility import load_data
import random
import numpy as np
import pandas as pd
from time import time


def generate_fake_sample(data, opts, substitute_num=2, field_weight={}):
    while True:
        temp = data[np.random.randint(len(data))][0:opts.sequence_length]
        if len(temp) < opts.sequence_length:
            gap = opts.sequence_length - len(temp)
            temp = np.array([0] * gap + temp)
        assert len(temp) == opts.sequence_length
        targets_to_avoid = set(temp)
        indices_to_avoid = set()
        substitute_index = np.random.randint(opts.sequence_length)
        substitute_target = np.random.randint(opts.vocabulary_size)
        for _ in range(substitute_num):
            while substitute_target in targets_to_avoid:
                substitute_target = np.random.randint(opts.vocabulary_size)
            targets_to_avoid.add(substitute_target)
            while substitute_index in indices_to_avoid:
                substitute_index = np.random.randint(opts.sequence_length)
            indices_to_avoid.add(substitute_index)
            temp[substitute_index] = substitute_target
        yield temp


def generate_discriminant_batch(data, opts, rate=0.5):
    data_index = 0
    fake_sample_generator = generate_fake_sample(data, opts)
    while True:
        batch = np.ndarray(shape=(opts.batch_size, opts.sequence_length))
        labels = np.ndarray(shape=(opts.batch_size, opts.num_classes))
        for i in xrange(opts.batch_size):
            target = np.zeros(opts.num_classes)
            if random.random() > rate:
                target[1] = 1.
                temp = data[data_index][-opts.sequence_length - 1:-1]
                if len(temp) < opts.sequence_length:
                    gap = opts.sequence_length - len(temp)
                    temp = np.array(temp + [0] * gap)
                assert len(temp) == opts.sequence_length
                batch[i] = temp
                labels[i] = target
                data_index = (data_index + 1) % len(data)
            else:
                target[0] = 1.
                batch[i] = fake_sample_generator.next()
                # batch[i] = np.random.choice(opts.vocabulary_size, opts.sequence_length)
                labels[i] = target

        yield batch, labels


class DiscriminantCat2Vec(Cat2Vec):

    def __init__(self, options, session, cate2id, id2cate, pre_trained_emb=None, trainable=True, pre_trained_path=None):
        self.pre_trained_emb = None
        self.trainable = trainable
        if pre_trained_path is not None:
            self.load_pre_trained(pre_trained_path)
        Cat2Vec.__init__(self, options, session, cate2id, id2cate)
        # self.build_graph()

    def load_pre_trained(self, path):
        self.pre_trained_emb = np.array(pd.read_csv(
            path, sep=',', header=None), dtype=np.float32)
        print('pre-trained shape', self.pre_trained_emb.shape)

    def build_graph(self):
        """Build the model graph."""
        opts = self._options
        first_indices, second_indices = \
            get_batch_pair_indices(opts.batch_size, opts.sequence_length)
        # print(first_indices.shape)
        # the following is just for example, base class should not include this
        # with self._graph.as_default():
        self.train_inputs = tf.placeholder(tf.int32,
                                           shape=[opts.batch_size,
                                                  opts.sequence_length])
        self.train_labels = tf.placeholder(tf.int32, shape=[opts.batch_size,
                                                            opts.num_classes])
        with tf.device('/cpu:0'):

            if self.pre_trained_emb is None:
                self.embeddings = tf.Variable(tf.random_uniform(
                    [opts.vocabulary_size, opts.embedding_size], -1.0, 1.0))
            else:
                if self.pre_trained_emb.shape == (opts.vocabulary_size, opts.embedding_size):
                    self.embeddings = tf.get_variable(name="embeddings",
                                                      shape=[
                                                          opts.vocabulary_size, opts.embedding_size],
                                                      dtype=tf.float32,
                                                      initializer=tf.constant_initializer(
                                                          self.pre_trained_emb),
                                                      trainable=self.trainable)
                    print('Inited by pre-trained embeddings')
                else:
                    print('pre_trained_emb shape', self.pre_trained_emb.shape)
                    print('vocabulary_size,embedding_size',
                          (opts.vocabulary_size, opts.embedding_size))
                    raise Exception('Error', 'pre_trained_emb size mismatch')

            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            encoded = sample_encoding(embed, opts.interaction_times,
                                      opts.batch_size, opts.sequence_length,
                                      opts.sequence_length, first_indices,
                                      second_indices, opts.gate_type,
                                      opts.norm_type)
            encoded = tf.reshape(embed, [opts.batch_size, -1])
            encoded = tf.concat(
                1, [encoded, tf.reshape(embed, [opts.batch_size, -1])])
            with tf.name_scope("output"):
                encoded_size = encoded.get_shape().as_list()[1]
                W, b = weight_bias([encoded_size, opts.num_classes], [
                                   opts.num_classes], bias_init=0.)
                scores = tf.matmul(encoded, W) + b
                self.predictions = tf.argmax(scores, 1, name="predictions")

            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    scores, tf.to_float(self.train_labels))
                self.loss = tf.reduce_mean(losses)

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(
                    self.predictions, tf.argmax(self.train_labels, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"), name="accuracy")

            # optimizer = \
            #     tf.train.GradientDescentOptimizer(opts.learning_rate)
            self.loss = tf.clip_by_value(self.loss, -10, 10)
            optimizer = tf.train.AdamOptimizer()
            self.train_operator = \
                optimizer.minimize(self.loss,
                                   gate_gradients=optimizer.GATE_NONE)
        tf.initialize_all_variables().run()
        print("Initialized")

    def train(self, batch_generator, num_steps):
        opts = self._options
        average_loss = 0.
        acc = 0.
        start = time()
        for step in xrange(num_steps):
            batch_inputs, batch_labels = batch_generator.next()
            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}
            _, loss, accuracy = self._session.run([self.train_operator,
                                                   self.loss,
                                                   self.accuracy],
                                                  feed_dict=feed_dict)
            average_loss += loss
            acc += accuracy
            if step % 1000 == 0:
                t = time() - start
                if step > 0:
                    average_loss /= 500
                    t /= 500
                    acc /= 500
                print("Average loss at step ", step, ": ", average_loss,
                      ' accuracy: ', acc, 'time', t)
                average_loss = 0
            if step % 1000 == 0:
                print('Eval at step ', step)
                self.eval_clustering()


def main():
    opts = Options()
    print('Loading data...')
    data, id2cate, cate2id, vocabulary_size = load_data(debug=True)
    opts.sequence_length = 24
    opts.vocabulary_size = vocabulary_size
    opts.norm_type = 'l2'
    opts.gate_type = 'mul'
    opts.batch_size = 32
    opts.embedding_size = 32
    opts.interaction_times = 3
    batch_generator = generate_discriminant_batch(data, opts)

    pre_trained_path = './data/ipinyou/pre_trained_embs_skip_cat_72746_32.csv'
    print('Building graph')
    with tf.Graph().as_default(), tf.Session() as session:
        discr_cat2vec = DiscriminantCat2Vec(opts, session, id2cate, cate2id,
                                            pre_trained_emb=None,
                                            trainable=True,
                                            pre_trained_path=pre_trained_path)
        print('Training model')
        discr_cat2vec.train(batch_generator, 100001)

if __name__ == "__main__":
    main()
