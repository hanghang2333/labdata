import numpy as np
import tensorflow as tf

zero = tf.constant(0, dtype=tf.int32)
gt = np.array([ [[0,1]]])
ground_truth = tf.constant(gt,dtype=tf.int32)

pd = np.array([ [[9,2]]])
predictions = tf.constant(pd,dtype=tf.float32)

true_bool = tf.not_equal(ground_truth, zero) #shape = (batch_size, sentence_length, y_tag_set) (2,3,5)
false_bool = tf.equal(ground_truth, zero) #shape = (batch_size, sentence_length, y_tag_set) (2,3,5)

true_bool_float = tf.to_float(true_bool) #(2,3,5)
false_bool_float = tf.to_float(false_bool) #(2,3,5)

true_prediction = tf.multiply(predictions, true_bool_float) #shape = (batch_size, sentence_length, y_tag_set) (2,3,5)
false_prediction = tf.multiply(predictions, false_bool_float)#shape = (batch_size, sentence_length, y_tag_set) (2,3,5)

tile_T = tf.tile(tf.expand_dims(true_prediction, -1), [1,1,1, tf.shape(false_prediction)[-1]])  # shape = (2,3,5,5)
tile_F = tf.tile(tf.expand_dims(false_prediction, -2), [1,1,tf.shape(true_prediction)[-1],1]) # shape = (2,3,5,5)

tile_a = tf.tile(tf.expand_dims(true_bool, -1), [1,1,1, tf.shape(ground_truth)[-1]]) # 2,3,5,5
tile_a_new = tf.to_float(tile_a)
print(tile_a_new.shape)
tile_b = tf.tile(tf.expand_dims(false_bool, -2), [1,1,tf.shape(ground_truth)[-1],1])  # 2,3,5,5
tile_b_new = tf.to_float(tile_b)

cartesian_product_index = tf.multiply(tile_a_new, tile_b_new) #shape = (2,3,5,5,1) 


loss = -tf.reduce_sum(
    tf.subtract(
        tf.multiply(tile_T, (cartesian_product_index)),
        tf.multiply(tile_F, (cartesian_product_index))
    )
)
result = tf.cast(loss, dtype=tf.float32)

count = tf.count_nonzero(cartesian_product_index)
c = tf.cast(count, dtype=tf.float32)
re_loss = tf.div(result, c)

print('here')
sess = tf.Session()
a = sess.run([tile_a_new])
print(a,a[0].shape)

a = sess.run([tile_b_new])
print(a,a[0].shape)
print(sess.run([cartesian_product_index]))
print(sess.run([count]))
print(sess.run([loss]))
