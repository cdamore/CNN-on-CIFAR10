from ops import *
import timeit
from iterator import DatasetIterator
from tensorflow.contrib.layers import flatten

def net(input, is_training, dropout_kept_prob):

  mu = 0
  sigma = 0.1

  # Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x6.
  conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name='conv1_W')
  conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
  conv1   = tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

  # Relu Activation.
  conv1 = tf.nn.relu(conv1)
  # Densenet-like connection: Convolutional. Input = 32x32x6. Output = 32x32x16.
  convs_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 9, 16), mean = mu, stddev = sigma), name='convs_W')
  convs_b = tf.Variable(tf.zeros(16), name='conv2_s')
  convs   = tf.nn.conv2d(tf.concat([input,conv1],axis=3), convs_W, strides=[1, 1, 1, 1], padding='SAME') + convs_b

  # Activation.
  convs = tf.nn.relu(convs)

  # Pooling. Input = 32x32x16. Output = 16x16x36.
  convs = tf.nn.max_pool(convs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  # Layer 3: Convolutional. Output = 10x10x16.
  conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 36), mean = mu, stddev = sigma), name='conv2_W')
  conv2_b = tf.Variable(tf.zeros(36), name='conv2_b')
  conv2   = tf.nn.conv2d(convs, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

  # Batch normalization
  conv2 = tf.layers.batch_normalization(conv2,training=is_training)

  # Activation.
  conv2 = tf.nn.relu(conv2)

  # Pooling. Input = 12x12x36. Output = 6x6x36.
  conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  # Flatten. Input = 6x6x36. Output = 1296.
  fc0 = flatten(conv2)

  #Dropout
  DropMask = (tf.to_float(tf.random_uniform((1,1296))<dropout_kept_prob))/dropout_kept_prob
  fc0 = tf.cond(is_training, lambda: fc0*DropMask, lambda: fc0)

  # Layer 4: Fully Connected. Input = 1296. Output = 120.
  fc1_W = tf.Variable(tf.truncated_normal(shape=(1296, 120), mean = mu, stddev = sigma), name='fc1_W')
  fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
  fc1   = tf.matmul(fc0, fc1_W) + fc1_b

  # Activation.
  fc1 = tf.nn.relu(fc1)

  # Layer 5: Fully Connected. Input = 120. Output = 84.
  fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name='fc2_W')
  fc2_b  = tf.Variable(tf.zeros(84), name='fc2_b')
  fc2    = tf.matmul(fc1, fc2_W) + fc2_b

  # Activation.
  fc2    = tf.nn.relu(fc2)

  # Layer 6: Fully Connected. Input = 84. Output = 10.
  fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma), name='fc3_W')
  fc3_b  = tf.Variable(tf.zeros(10), name='fc3_b')
  logits = tf.matmul(fc2, fc3_W) + fc3_b

  return logits

def train(x_train, y_train, batch_size):
  # reset graph
  tf.reset_default_graph()

  EPOCHS = 20
  N_BATCHES = 250 # 50,000 / 200

  # true
  isTrain = tf.placeholder(tf.bool)

  # training variables
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  y = tf.placeholder(tf.int32, (None))
  one_hot_y = tf.one_hot(y, 10)
  # learning rate
  rate = 0.001
  # get output from cnn
  logits = net(x, isTrain, 1) # no dropout_kept_prob

  # init saver
  saver = tf.train.Saver(max_to_keep=0)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
  loss_operation = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate = rate)
  grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation = optimizer.apply_gradients(grads_and_vars)

  # init dataset iterator
  cifar10_train = DatasetIterator(x_train, y_train, batch_size)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training...")
    global_step = 0
    for i in range(EPOCHS):
        for iteration in range(N_BATCHES):
            # get next batch
            batch_x, batch_y = cifar10_train.get_next_batch()
            _ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, isTrain: True})
            global_step += 1

        print("EPOCH {} ...".format(i+1))
        print()
        # save latest model
        saver.save(sess, 'ckpt/net', global_step=i)

    print("Model saved")

def test(cifar10_test_images):
  tf.reset_default_graph()

  # true
  isTest = tf.placeholder(tf.bool)
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  y = tf.placeholder(tf.int32, (None))
  one_hot_y = tf.one_hot(y, 10)
  # learning rate
  rate = 0.001
  # get output from cnn
  logits = net(x, isTest, 1) # no dropout_kept_prob

  # init saver
  saver = tf.train.Saver(max_to_keep=0)

  # pipeline
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
  loss_operation = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate = rate)
  grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation = optimizer.apply_gradients(grads_and_vars)

  with tf.Session() as sess:
      # Load saved model from ckpt folder
      saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
      # Get the predicted classes of the test images
      yp_test = sess.run(tf.argmax(logits, axis=1), feed_dict={x: cifar10_test_images, isTest: True})

  return yp_test
