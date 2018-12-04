import tensorflow as tf
import reader

from datetime import datetime
import os.path
import time
import numpy as np

# Tensorflow

NUM_CLASSES = 10

class CNNModel():
    def __init__(self):
        # internal setting
        self._optimizer = tf.train.AdamOptimizer()
        
        # config
        self._batch_size = 128
        self._max_steps = 60000

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')

            tf.add_to_collection('losses', weight_decay)
        return var
        
    def _build_graph(self, images):
        bias_initializer = tf.constant_initializer(0.0)

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 3, self._batch_size], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [self._batch_size], initializer=bias_initializer, dtype=tf.float32)
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, self._batch_size, self._batch_size], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [self._batch_size], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [self._batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
            biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
            biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
            biases = tf.get_variable('biases', [NUM_CLASSES], initializer=bias_initializer, dtype=tf.float32)
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        return softmax_linear

    def loss(self, logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self, data, session):
        labels, images = reader.cifar10_train_iterator(data, self._batch_size)
        logits = self._build_graph(images)
        loss_op = self.loss(logits, labels)
        optimize_op = self._optimizer.minimize(loss_op)
        saver = tf.train.Saver(tf.all_variables())
        session.run(tf.initialize_all_variables())

        # バッチ化するにあたってキューに貯めたデータを評価毎に取り出す処理を開始する
        tf.train.start_queue_runners(sess=session)

        for step in range(self._max_steps):
            # ミニバッチごとの処理
            start_time = time.time()
            loss, _ = session.run([loss_op, optimize_op])
            duration = time.time() - start_time

            assert not np.isnan(loss), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = self._batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
                print (format_str % (datetime.now(), step, loss, examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == self._max_steps:
                checkpoint_path = os.path.join('model', 'model.ckpt')
                saver.save(session, checkpoint_path, global_step=step)

    def evaluate(self, data, session):
        # モデルを読み込む前にあらかじめ必要な変数を定義しておく必要がある
        labels, images = reader.cifar10_eval_iterator(data, self._batch_size)
        logits = self._build_graph(images)
        top_1_op = tf.nn.in_top_k(logits, labels, 1)

        # Load model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model')
        saver.restore(session, ckpt.model_checkpoint_path)

        true_count = 0
        tf.train.start_queue_runners(sess=session)
        for i in range(int(len(data[0]) / self._batch_size)):
            predictions = session.run(top_1_op)
            true_count += np.sum(predictions)

        precision = true_count / self._batch_size # len(data[0])
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))


    def predict(self, data, session):
        # モデルを読み込む前にあらかじめ必要な変数を定義しておく必要がある
        labels, images = reader.cifar10_eval_iterator(data, self._batch_size)
        logits = self._build_graph(images)
        top_1_op = tf.nn.top_k(logits, 1)

        # Load model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model')
        saver.restore(session, ckpt.model_checkpoint_path)

        tf.train.start_queue_runners(sess=session)
        for i in range(int(len(data[0]) / self._batch_size)):
            predictions, label = session.run([top_1_op, labels])
            print(np.reshape(predictions.indices, (self._batch_size, )))
            print(label)

def main():
    print("start CNN")
    train_data, test_data = reader.cifar10_raw_data("cifar-10-batches-py")

    # 学習
    with tf.Graph().as_default():
        model = CNNModel()
        session = tf.Session()
        model.train(train_data, session)

    # 推論
    with tf.Graph().as_default():
        model = CNNModel()
        session = tf.Session()
        model.predict(test_data, session)

    # 評価
    with tf.Graph().as_default():
        model = CNNModel()
        session = tf.Session()
        model.evaluate(test_data, session)

if __name__ == '__main__':
    main()
Raw
