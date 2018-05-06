import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from model import HierarchicalAttention
from utils import parser, read_tfrecord

shuffle_size = 100000
vocab_size = 71825 + 1
sent_num = 20
word_num = 20
batch_size = 64
embedding_dim = 100
hidden_size = 50
num_classes = 5
use_pretrained = False
initializer = tf.random_uniform_initializer()
starter_learning_rate = 0.001
model_path = 'train/'
epochs = 4

tf.reset_default_graph()

train_data_dir = './train.tfrecord'
eval_data_dir = './valid.tfrecord'
test_data_dir = './test.tfrecord'

train_itr, train_dataset, train_doc, train_label = read_tfrecord(train_data_dir, 
                                                                 parser, 
                                                                 shuffle_size, 
                                                                 batch_size, 
                                                                 sent_num, 
                                                                 word_num)

eval_itr, eval_dataset, eval_doc, eval_label = read_tfrecord(eval_data_dir, 
                                                             parser, 
                                                             shuffle_size, 
                                                             batch_size, 
                                                             sent_num, 
                                                             word_num)

test_itr, test_dataset, test_doc, test_label = read_tfrecord(test_data_dir, 
                                                             parser, 
                                                             shuffle_size, 
                                                             batch_size, 
                                                             sent_num, 
                                                             word_num)


train_init_op = train_itr.make_initializer(train_dataset)
eval_init_op = eval_itr.make_initializer(eval_dataset)
test_init_op = test_itr.make_initializer(test_dataset)

with tf.device('/gpu:0'):
    X = tf.placeholder(tf.int64, [None, sent_num, word_num])
    y = tf.placeholder(tf.int64, [None, num_classes])
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.95)
    
    model = HierarchicalAttention(vocab_size, 
                                  sent_num, 
                                  word_num, 
                                  embedding_dim, 
                                  hidden_size, 
                                  num_classes, 
                                  use_pretrained, 
                                  initializer)
    
    train_y_hat = model.build_graph(X, reuse=False)
    
    loss = tf.losses.log_loss(y, train_y_hat)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    acc_y_hat = model.build_graph(X, reuse=True)
    prediction = tf.argmax(acc_y_hat, 1)
    
    train_accuracy = tf.metrics.accuracy(tf.argmax(y, 1), prediction)
    eval_accuracy = tf.metrics.accuracy(tf.argmax(y, 1), prediction)
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', train_accuracy[1])
    
    merged = tf.summary.merge_all()
    
saver = tf.train.Saver(tf.global_variables())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    writer = tf.summary.FileWriter(model_path, sess.graph)
    
    for epoch in range(epochs):
        sess.run(train_init_op)
        
        while True:
            try:
                step = sess.run(global_step)
                
                _doc, _label = sess.run([train_doc, train_label])
                _, _loss, _merged = sess.run([optimizer, loss, merged], feed_dict={X: _doc, y: _label})
                _acc = sess.run(train_accuracy, feed_dict={X: _doc, y: _label})
                
                writer.add_summary(_merged, step)
                
                if (step > 0) and (step % 500 == 0):
                    print('step {}, loss: {}, train_accuracy: {}'.format(step, _loss, _acc[1]))
                    
            except tf.errors.OutOfRangeError:
                break
                
        sess.run(eval_init_op)
        eval_acc = []
        
        while True:
            try:
                _doc, _label = sess.run([eval_doc, eval_label])
                _acc = sess.run(eval_accuracy, feed_dict={X: _doc, y: _label})
                
                eval_acc.append(_acc[1])
                
            except tf.errors.OutOfRangeError:
                break
                
        print('epoch {}, eval_acc: {}'.format(epoch, np.mean(eval_acc)))
    
        saver.save(sess, model_path + 'epoch_' + str(epoch) + '.ckpt', global_step=sess.run(global_step))
    
        print('Model is saved.')
        
    sess.run(test_init_op)    
    test_acc = np.array([], dtype='int32')
    true_label = np.array([], dtype='int32')
        
    while True:
        try:
            _doc, _label = sess.run([test_doc, test_label])
            _acc = sess.run(prediction, feed_dict={X: _doc, y: _label})
            
            test_acc = np.concatenate([test_acc, _acc])
            true_label = np.concatenate([true_label, np.argmax(_label, 1)])
            
        except tf.errors.OutOfRangeError:
            break
    
    print('Test accuracy: {}'.format(accuracy_score(true_label, test_acc)))

print("End.")