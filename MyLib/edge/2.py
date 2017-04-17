import tensorflow as tf

t_0 = tf.Variable(50)       # 0차원 텐서

t_1 = tf.Variable(["apple", "peach", "grape"])      # 1차원 텐서

t_2 = tf.Variable([[True, False, False],        # 2차원 텐서
       [False, False, True],
       [False, True, False]   ])

t_3 = tf.Variable([     # 3차원 텐서
    [[0,0],[0,1],[0,2]],
    [[1,0], [1,1], [1,1]],
    [[2,0], [2,1], [2,2]]
])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print(sess.run(t_0))
print(sess.run(t_1))
print(sess.run(t_2))
print(sess.run(t_3))

print("====")
print(tf.shape(t_1))