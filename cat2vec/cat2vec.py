import tensorflow as tf
import numpy as np
import os
import random
from utility import *


def generate_batch_skip_gram(data, batch_size, num_skips):
    '''
    Batch generator for Skip-gram
    '''
    data_index = 0
    assert batch_size % num_skips == 0
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    while True:
        for i in range(batch_size // num_skips):
            span = len(data[data_index])
            # target label at the center of the buffer
            label_index = random.randint(0, span - 1)
            singe_data = data[data_index]
            targets_to_avoid = [label_index]
            target = label_index
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = singe_data[label_index]
                labels[i * num_skips + j, 0] = singe_data[target]
            data_index = (data_index + 1) % len(data)
        yield batch, labels


def generate_batch_cbow(data, batch_size, num_skips):
    '''
    Batch generator for CBOW (Continuous Bag of Words).
    '''
    data_index = 0
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    while True:
        for i in range(batch_size):
            span = len(data[data_index])
            label_index = random.randint(0, span - 1)
            singe_data = data[data_index]
            labels[i, 0] = singe_data[label_index]

            targets_to_avoid = [label_index]
            target = label_index
            sample = []
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                sample.append(singe_data[target])
            batch[i] = sample
            data_index = (data_index + 1) % len(data)
        yield batch, labels


class Options(object):
    """Options used by  the model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.embedding_size = 32

        # The initial learning rate.
        self.learning_rate = 1.

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = 100

        # Number of examples for one training step.
        self.batch_size = 128

        # Number of class
        self.num_classes = 2

        self.sequence_length = 25

        self.interaction_times = 3

        # self.k = self.sequence_length

        self.vocabulary_size = 10000000

        # Where to write out summaries.
        self.save_path = './'

        self.dataset = 'ipinyou'

        self.eval_data = './data/' + self.dataset + '/questions.txt'

        self.architecture = 'skip-gram'

        self.num_skips = 4
        self.field_cate_indices_path = \
            './data/ipinyou/field_cates_index_not_aligned.csv'
        # for negative sampling
        self.valid_size = 16
        self.valid_window = 100
        self.valid_examples = np.random.choice(
            self.valid_window, self.valid_size, replace=False)
        self.num_sampled = 64    # Number of negative examples to sample.


class Cat2Vec(object):
    """
        Base Class for Cat2Vec
        Based on tensorflow/embedding/word2vec.py
    """

    def __init__(self, options, session, id2cate, cate2id):
        self._options = options
        self._cate2id = cate2id
        self._id2cate = id2cate
        self._session = session
        self.fields_index = None
        # self.embeddings = None
        self._read_analogies()
        self.build_graph()
        self.build_eval_graph()

    def _read_analogies(self):
        """Reads through the analogy question file.
        Returns:
          questions: a [n, 4] numpy array containing the analogy question's
                     word ids.
          questions_skipped: questions skipped due to unknown words.
        """
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(" ")
                # print words
                ids = [self._cate2id.get(w.strip()) for w in words]
                # print ids
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", self._options.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        questions = np.array(questions, dtype=np.int32)
        self._analogy_questions = questions
        self._target_field = np.array(
            list(set(questions[:, 3])), dtype=np.int32)
        np.random.shuffle(self._analogy_questions)

    def build_eval_graph(self):
        """Build the evaluation graph."""
        # Eval graph
        opts = self._options

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        nemb = tf.nn.l2_normalize(self.embeddings, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, tf.gather(
            nemb, self._target_field), transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 5)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, opts.vocabulary_size))

        field_cates = tf.placeholder(dtype=tf.int32)
        field_embs = tf.gather(self.embeddings, field_cates)
        center_point = tf.reduce_mean(field_embs, 0)
        avg_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            tf.pow(tf.sub(center_point, field_embs), 2), 1)), 0)

        self._avg_distance = avg_distance
        self._field_cates = field_cates
        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        # Properly initialize all variables.
        # tf.initialize_all_variables().run()

        # self.saver = tf.train.Saver()

    def build_graph(self):

        opts = self._options
        with tf.device('/cpu:0'):
            if opts.architecture == 'skip-gram':
                self.train_inputs = tf.placeholder(tf.int32,
                                                   shape=[opts.batch_size])
            elif opts.architecture == 'cbow':
                self.train_inputs = tf.placeholder(
                    tf.int32, shape=[opts.batch_size, opts.num_skips])
            self.train_labels = tf.placeholder(tf.int32,
                                               shape=[opts.batch_size, 1])
            valid_dataset = tf.constant(opts.valid_examples, dtype=tf.int32)
            self.embeddings = tf.Variable(
                tf.random_normal([opts.vocabulary_size,
                                  opts.embedding_size],
                                 stddev=1.0 / np.sqrt(opts.embedding_size)
                                 ))

            weights = tf.Variable(
                tf.truncated_normal([opts.vocabulary_size,
                                     opts.embedding_size],
                                    stddev=1.0 / np.sqrt(opts.embedding_size)
                                    ))
            biases = tf.Variable(tf.zeros([opts.vocabulary_size]))

            if opts.architecture == 'skip-gram':
                embed = tf.nn.embedding_lookup(self.embeddings,
                                               self.train_inputs)
            elif opts.architecture == 'cbow':
                embed = tf.zeros([opts.batch_size, opts.embedding_size])
                for j in range(opts.num_skips):
                    embed += tf.nn.embedding_lookup(self.embeddings,
                                                    self.train_inputs[:, j])

            if opts.architecture == 'skip-gram':
                losses = tf.nn.nce_loss(weights, biases, embed,
                                        self.train_labels,
                                        opts.num_sampled, opts.vocabulary_size)
            elif opts.architecture == 'cbow':
                losses = tf.nn.sampled_softmax_loss(weights, biases, embed,
                                                    self.train_labels,
                                                    opts.num_sampled,
                                                    opts.vocabulary_size)

            self.loss = tf.reduce_mean(losses)

            # Construct the SGD optimizer using a learning rate of 1.0.
            self.optimizer = tf.train.GradientDescentOptimizer(opts.learning_rate)\
                               .minimize(self.loss)

            # norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),
            #                              1, keep_dims=True))
            # self.normalized_embeddings = self.embeddings / norm
            # valid_embeddings = tf.nn.embedding_lookup(
            #     self.normalized_embeddings, valid_dataset)
            # self.similarity = tf.matmul(
            #     valid_embeddings, self.normalized_embeddings, transpose_b=True)
            # Add variable initializer.
            tf.initialize_all_variables().run()
            self.saver = tf.train.Saver()

    def train(self, batch_generator, num_steps):
        """Train the model."""
        # opts = self._options
        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = batch_generator.next()
            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}
            _, loss_val = self._session.run([self.optimizer, self.loss],
                                            feed_dict=feed_dict)
            average_loss += loss_val

            if step % 50000 == 0:
                if step > 0:
                    average_loss /= 50000
                # print(eb[104])
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500
        # steps)
            if step % 100000 == 0:
                print('Eval at step ', step)
                self.eval_clustering()
                # self.eval()
                # sim = self.similarity.eval()
                # for i in xrange(opts.valid_size):
                #     valid_word = self._id2cate[opts.valid_examples[i]]
                #     top_k = 8  # number of nearest neighbors
                #     nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #     log_str = "Nearest to %s:" % valid_word
                #     for k in xrange(top_k):
                #         close_word = self._id2cate[nearest[k]]
                #         log_str = "%s %s," % (log_str, close_word)
                #     print(log_str)

    def plot(self):
        """Plot the low demension embeddings"""
        pass

    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]),
                                     opts.vocab_counts[i]))

    def _predict(self, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx

    def eval_clustering(self):
        opts = self._options
        if self.fields_index is None:
            if opts.field_cate_indices_path is None:
                opts.field_cate_indices_path =\
                    './data/ipinyou/field_cates_index_not_aligned.csv'
            self.fields_index = {}
            f = open(opts.field_cate_indices_path, 'r')
            for line in f.readlines():
                field_name, indices = line.strip().split('\t')
                indices = np.array([int(i) for i in indices.split(',')])
                self.fields_index[field_name] = indices
        avg_distacnes = []

        for field_name in self.fields_index.keys():
            field_indices = self.fields_index.get(field_name)
            field_feed_dict = {self._field_cates: field_indices}
            distance = self._session.run([self._avg_distance],
                                         feed_dict=field_feed_dict)
            avg_distacnes.extend(distance)
            print('avg distance for field:', field_name, ':', distance)
        print('avg distance for all field:', np.mean(avg_distacnes))

    def eval(self):
        """Evaluate analogy questions and reports accuracy."""

        # How many questions we get right at precision@1.
        correct = 0

        total = self._analogy_questions.shape[0]
        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                if sub[question, 3] in idx[question]:
                    # print(sub[question, 3], idx[question])
                    correct += 1

        print()
        print("Eval %4d/%d accuracy @ top5= %4.1f%%" % (correct, total,
                                                        correct * 100. / total)
              )

    def analogy(self, c0, c1, c2):
        """Predict category c3 as in c0:c1 vs c2:c3."""
        cid = np.array([[self._cate2id.get(c) for c in [c0, c1, c2]]])
        idx = self._predict(cid)
        for c in [self._id2cate[i] for i in idx[0, :]]:
            if c not in [c0, c1, c2]:
                return c
        return "unknown"

    def nearby(self, categories, num=20):
        """Prints out nearby categories given a list of categories."""
        ids = np.array([self._cate2id.get(x) for x in categories])
        vals, idx = self._session.run(
            [self._nearby_val, self._nearby_idx],
            {self._nearby_categories: ids})
        for i in xrange(len(categories)):
            print("\n%s\n=====================================" %
                  (categories[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self._id2cate[neighbor], distance))


def main():
    opts = Options()
    opts.architecture = 'cbow'
    print('Loading data...')
    data, id2cate, cate2id, vocabulary_size = load_data(debug=False)
    opts.vocabulary_size = vocabulary_size
    print(opts.vocabulary_size)

    if opts.architecture == 'skip-gram':
        batch_generator = generate_batch_skip_gram(data, opts.batch_size,
                                                   opts.num_skips)
    elif opts.architecture == 'cbow':
        batch_generator = generate_batch_cbow(data, opts.batch_size,
                                              opts.num_skips)
    print('Building graph')
    with tf.Graph().as_default(), tf.Session() as session:
        cat2vec = Cat2Vec(opts, session, id2cate, cate2id)
        print('Training model')
        cat2vec.train(batch_generator, 1000001)
        # print('Evaluating model')
        # cat2vec.eval()
        # cat2vec.eval_clustering()

if __name__ == "__main__":
    main()
