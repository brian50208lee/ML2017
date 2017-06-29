# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time, sys
import numpy as np
import tensorflow as tf
from random import randint
from sklearn.model_selection import train_test_split
import util




def map_str_to_float(lt):
  return [float(i) for i in lt]

def convert_label_to_list(value):
  lt = [0.0, 0.0, 0.0]
  lt[value] = 1.0
  return lt

def convert_label_to_str(label_dict, prediction):
  return [label_dict[i] for i in prediction]

def read_data(test_id_path, validation_size=0.2):
  x, x_test = util.load_data()
  y = util.load_target().tolist()
  test_id = list()

  # normalize
  # 0, 48, 67, 68, 74, 175, 192 columns > 1.0, < 0.0
  need_normalize = [0, 48, 67, 68, 74, 175, 192]
  for column in need_normalize:
    x[:,column] = (x[:,column]  - x[:,column] .mean()) / x[:,column] .std()
    x_test[:,column] = (x_test[:,column] - x_test[:,column].mean()) / x_test[:,column].std()
  x = x.tolist()
  x_test = x_test.tolist()

  i = 0
  for line in open(test_id_path, 'r'):
    if i == 0:
      i += 1
      continue
    test_id.append(line.strip().split(',')[0])
  for i, label in enumerate(y):
    lt = [0.0, 0.0, 0.0]
    lt[label] = 1.0
    y[i] = lt
  x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size)
  return x_train, x_validation, y_train, y_validation, x_test, test_id


def next_batch(x_train, y_train, batch_size):
  batch = list()
  rand = randint(0, len(x_train) - batch_size)
  batch.append(x_train[rand: rand + batch_size])
  batch.append(y_train[rand: rand + batch_size])
  return batch


# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations


log_dir = './summaries'

def train():
  # Create a multilayer model.
  validation_size = 0.2
  x_train, x_validation, y_train, y_validation, x_test, test_id = read_data(sys.argv[3], validation_size)
  dim = len(x_train[0])
  classes = 3
  label_dict = {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}
  
  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, dim], name='x-input')
    y = tf.placeholder(tf.float32, [None, classes], name='y-input')

  epochs = 100
  batch_size = len(x_train) // 20
  lr = 1e-3
  DNN_layers = 5
  neuron = 32
  prob = 0.5
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
  best_cross_entropy = 0.565

  hidden_layers = dict()
  dropped = dict()

  hidden_layers[1] = nn_layer(x, dim, neuron, 'layer1')
  dropped[1] = tf.nn.dropout(hidden_layers[1], keep_prob)

  for i in range(2, DNN_layers):
    hidden_layers[i] = nn_layer(dropped[i - 1], neuron, neuron, 'layer' + str(i))
    dropped[i] = tf.nn.dropout(hidden_layers[i], keep_prob)

  # Do not apply softmax activation yet, see below.
  output = nn_layer(dropped[DNN_layers - 1], neuron, classes, 'layer' + str(DNN_layers), act=tf.identity)
  prediction_prob = tf.nn.softmax(output)
  prediction = tf.argmax(output, axis=1)

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  init = tf.global_variables_initializer()
  sess = tf.Session()
  saver = tf.train.Saver()


  # Merge all the summaries and write them out
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(log_dir + '/test')

  # Train the model, and also write summaries.
  # Every epoch, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  # sess.run(init)
  # for epoch in range(1, epochs + 1):
  #   start = time.time()
  #   train_cross_entropy = 0.0
  #   # Record train set summaries, and train
  #   # Record execution stats
  #   for i in range((len(x_train) // batch_size) + 1):
  #     step = (epoch - 1) * ((len(x_train) // batch_size) + 1) + i
  #     batch = next_batch(x_train, y_train, batch_size)
  #     summary, _, train_c_e = sess.run([merged, train_step, cross_entropy], feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
  #     train_writer.add_summary(summary, step)
  #     train_cross_entropy += train_c_e
  #   train_cross_entropy /= ((len(x_train) // batch_size) + 1)

  #   # Record summaries and test-set accuracy
  #   summary, acc, validation_cross_entropy = sess.run([merged, accuracy, cross_entropy], feed_dict={x:x_validation, y:y_validation, keep_prob: 1.0})
  #   test_writer.add_summary(summary, step)

  #   end = time.time()
  #   print('{}\t{}\t{}\t{}\t{} seconds'.format(epoch, train_cross_entropy, validation_cross_entropy, acc, end - start))

  #   if validation_cross_entropy < best_cross_entropy:
  #     best_cross_entropy = validation_cross_entropy
  #     print('save model with {}'.format(best_cross_entropy))
  #     path = saver.save(sess, './' + str(best_cross_entropy) + '.ckpt')

  # train_writer.close()
  # test_writer.close()

  # saver.restore(sess, './' + str(best_cross_entropy) + '.ckpt')
  # # train on validation set
  # for epoch in range(1, int(validation_size * epochs) + 1):
  #   start = time.time()
  #   train_cross_entropy = 0.0
  #   # Record train set summaries, and train
  #   # Record execution stats
  #   summary, _, train_cross_entropy = sess.run([merged, train_step, cross_entropy], feed_dict={x:x_validation, y:y_validation, keep_prob:0.5})

  #   end = time.time()
  #   print('{}\t{}\t{} seconds'.format(epoch, train_cross_entropy, end - start))
  # path = saver.save(sess, './best_model.ckpt')

  saver.restore(sess, './best_model.ckpt')
  # predict
  test_prediction = convert_label_to_str(label_dict, sess.run(prediction, feed_dict={x: x_test, keep_prob: 1.0}))
  test_prediction_prob = sess.run(prediction_prob, feed_dict={x: x_test, keep_prob: 1.0})
  output = open(sys.argv[4], 'w')
  output.write('id,status_group\n')
  for i in range(len(test_prediction)):
    output.write('{},{}\n'.format(test_id[i], test_prediction[i]))
    output.flush()
  output.close()


def main(_):
  # if tf.gfile.Exists(log_dir):
  #   tf.gfile.DeleteRecursively(log_dir)
  # tf.gfile.MakeDirs(log_dir)
  train()


if __name__ == '__main__':
  tf.app.run()