import tensorflow as tf
import numpy as np

sess=tf.Session()    

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('test_dla_pindzi.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
feed_dict ={X:[[1,2,3,4,5,6]]}

first_hidden_layer = graph.get_tensor_by_name("1/first_hidden_layer:0")
second_hidden_layer = graph.get_tensor_by_name("2/second_hidden_layer:0")
third_hidden_layer = graph.get_tensor_by_name("3/third_hidden_layer:0")
output_layer  = graph.get_tensor_by_name("output/output_layer:0")

print(sess.run(output_layer,feed_dict))