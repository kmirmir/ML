import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

add = tf.add(X, Y)
mul = tf.multiply(X, Y)

# step 1: node 선택
add_hist = tf.contrib.deprecated.scalar_summary("add_scalar", add)
mul_hist = tf.contrib.deprecated.scalar_summary("mul_scalar", mul)

# step 2: summary 통합. 두 개의 코드 모두 동작.
merged = tf.summary.merge_all()
# merged = tf.merge_summary([add_hist, mul_hist])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # step 3: writer 생성
    writer = tf.summary.FileWriter("./board/sample", sess.graph)

    for step in range(100):
        # step 4: 노드 추가
        summary = sess.run(merged, feed_dict={X: step * 2.0, Y: 3.0})
        writer.add_summary(summary, step)

# step 5: 콘솔에서 명령 실행
# tensorboard --logdir=./board/sample

