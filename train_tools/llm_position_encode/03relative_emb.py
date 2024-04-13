# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2023/12/16 12:33
# @Author  : acedar
# @FileName: 03relative_emb.py
# ----------------------------

import tensorflow as tf


def _generate_relative_positions_matrix(length, max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat = range_mat - tf.transpose(range_mat)
    else:
        distance_mat = tf.expand_dims(tf.range(-length+1, 1, 1), 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length, depth,
                                            max_relative_position, name,
                                            cache=False):
    """Generates tensor of size [1 if cache else length, length, depth]."""
    with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(
            length, max_relative_position, cache=cache)
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings

