import tensorflow as tf
import os


saver = tf.train.Saver()

with tf.Session() as sess:
    initial_step = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print(initial_step)
    # for step in range(initial_step, 2001):
    #     sess.run()