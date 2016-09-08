import tensorflow as tf
import numpy as np


def weight_bias(W_shape, b_shape, bias_init=0.):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b


def dense_layer(x, W_shape, b_shape, activation):
    W, b = weight_bias(W_shape, b_shape)
    return activation(tf.matmul(x, W) + b)


def flat_highway_gate_layer(first, second, carry_bias=0.,
                            activation=tf.tanh):
    X_shape = tf.shape(first)
    first = tf.reshape(first, [X_shape[0] * X_shape[1], X_shape[2]])
    second = tf.reshape(second, [X_shape[0] * X_shape[1], X_shape[2]])
    first_second = tf.concat(1, [first, second])
    W_T_shape = [X_shape[2], X_shape[2]]
    W_H_shape = [X_shape[2] * 2, X_shape[2]]
    b_shape = [X_shape[2]]
    W_T, b_T = weight_bias(W_T_shape, b_shape)
    # W_T_2, b_T_2 = weight_bias(W_T_shape, b_shape)
    H = dense_layer(first_second, W_H_shape, b_shape, activation)
    x = tf.add(first, second)
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
    # T2 = tf.sigmoid(tf.matmul(second, W_T_2) +
    #                 b_T_2, name='transform_gate_2')
    C = tf.sub(1.0, T, name="carry_gate")
    y = tf.add(tf.mul(H, T), tf.mul(x, C))  # y = (H * T) + (x * C)
    return y.reshape(X_shape)


def get_pair_indices(sequence_length):
    """get indices for each pair in a sample x"""
    pair_indices = []
    for i in range(sequence_length):
        for j in range(i + 1, sequence_length):
            pair_indices.append([i, j])
    return pair_indices


def get_batch_pair_indices(batch_size, sequence_length):
    """get the indices for a batch"""
    indices = get_pair_indices(sequence_length)
    comb_num = len(indices)
    batch_indices = []
    for i in range(batch_size):
        for j in range(len(indices)):
            batch_indices.append([[i, indices[j][0]], [i, indices[j][1]]])
    batch_indices = np.array(batch_indices)
    return (batch_indices[:, 0].reshape(batch_size, comb_num, 2),
            batch_indices[:, 1].reshape(batch_size, comb_num, 2))


def _gate(c1, c2, gate_type='sum'):
    """pair interaction method"""
    if gate_type == 'sum':
        return tf.add(c1, c2)
    if gate_type == 'mul':
        return tf.mul(c1, c2)
    if gate_type == 'avg':
        return tf.mul(tf.fill(tf.shape(c2), 0.5), tf.add(c1, c2))
    if gate_type == 'highway':
        return flat_highway_gate_layer(c1, c2)
    if gate_type == 'sum_tanh':
        return tf.tanh(tf.add(c1, c2))
    if gate_type == 'mul_tanh':
        return tf.tanh(tf.mul(c1, c2))
    if gate_type == 'p_norm':
        p = 2
        return tf.pow(tf.add([tf.pow(c1, p), tf.pow(c2, p)]), 1 / p)


def _norm(interactions, norm_type='l2'):
    """metrics for max-pooling"""
    if norm_type == 'l2':
        return tf.reduce_sum(tf.square(interactions), 2)
    if norm_type == 'l1':
        return tf.reduce_sum(tf.abs(interactions), 2)
    if norm_type == 'avg':
        return tf.reduce_sum(interactions, 2)


def gather_nd(params, indices):
    """
        just work for 3D tensor with 2D indices
        ref: cruvadom's implementation for gather_nd with some modification,
             https://github.com/tensorflow/tensorflow/issues/206
        e.g:
            params = [[[1,3],[2,2]]]
            indices = [[0,1]]
            gather_nd(params, indices) returns [[[2,2]]]
    """
    shape = params.get_shape().as_list()
    if type(indices) is tf.Tensor:
        indices_shape = indices.get_shape().as_list()
    elif type(indices) is np.ndarray:
        indices_shape = indices.shape
    else:
        raise ValueError("indices should be np.ndarray or tf.Tensor")

    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1)
                   for i in range(0, rank)]
    # print(multipliers)
    indices_unpacked = tf.unpack(tf.transpose(
        indices, [rank - 1] + range(0, rank - 1)))
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    flat_indices = tf.expand_dims(flat_indices, 2)
    flat_indices = tf.tile(flat_indices, [1, 1, shape[-1]])
    flat_indices_shape = flat_indices.get_shape().as_list()
    # we want output[i] = params[i,indices[i]]
    # index_mask = tf.reshape(tf.one_hot(indices, 3), [-1,3])
    # output = tf.reduce_sum(params * index_mask,1)
    # print(flat_indices_shape)
    indices_3d = tf.constant(flat_indices_shape[0] * flat_indices_shape[1] * range(shape[-1]), shape=flat_indices_shape)
    flat_indices = tf.add(tf.to_int32(flat_indices), indices_3d)
    flat_indices = tf.reshape(flat_indices, [-1])
    gathered = tf.gather(flat_params, flat_indices)
    # return flat_indices
    return tf.reshape(gathered, [shape[0], indices_shape[1], shape[-1]])


def pair_wise_interaction_and_max_pooling(X, batch_size, k,
                                          sequence_length,
                                          first_indices,
                                          second_indices,
                                          gate_type='sum',
                                          norm_type='l2'
                                          ):
    """
        input X: embedding inputs or max-pooling outputs
        output: pair wise interactions and max-pooling results
    """
    first_element = gather_nd(X, first_indices)
    second_element = gather_nd(X, second_indices)
    interactions = _gate(first_element, second_element, gate_type)
    norms = _norm(interactions, norm_type)
    _, max_indices = tf.nn.top_k(norms, k)
    pooling_indices = _get_max_pooling_indices(batch_size, max_indices, k)
    # print(pooling_indices.get_shape().as_list())
    pooling_rst = gather_nd(interactions, pooling_indices)
    return pooling_rst


def sample_encoding(X, interaction_times, batch_size, k,
                    sequence_length,
                    first_indices,
                    second_indices,
                    gate_type='sum',
                    norm_type='l2'):
    """
        input X: embedding inputs or max-pooling outputs
        output: Full connection rst after interactions and max-pooling
    """
    rst = X
    for i in range(interaction_times):
        with tf.name_scope("interaction_and_pooling_layer_%d" % i):
            rst = pair_wise_interaction_and_max_pooling(
                rst, batch_size, k, sequence_length,
                first_indices, second_indices)

    return full_connection(rst)


def full_connection(X):
    """
        Input: max-pooling results
        Output: concat rst or fc rst
    """
    X_shape = X.get_shape().as_list()
    rst = tf.reshape(X, [X_shape[0], X_shape[1] * X_shape[2]])
    return rst


def _get_max_pooling_indices(batch_size, indices, k):
    """
        refers from tianyao's code
        get the indices for the max-pooling
    """
    batch_indices = tf.range(batch_size)
    batch_indices = tf.expand_dims(batch_indices, 1)
    batch_indices = tf.expand_dims(batch_indices, 2)
    batch_indices = tf.tile(batch_indices, [1, k, 1])
    indices = tf.expand_dims(indices, 2)
    return tf.concat(2, [batch_indices, indices])
