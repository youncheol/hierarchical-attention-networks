import tensorflow as tf
import numpy as np

class HierarchicalAttention:
    def __init__(self, vocab_size, sent_num, word_num, embedding_dim, hidden_size, num_classes, use_pretrained, initializer):
        self.vocab_size = vocab_size
        self.sent_num = sent_num
        self.word_num = word_num
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.initializer = initializer
        
        
    def word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            embedding_W = tf.get_variable('embedding_W',
                                          shape=[self.vocab_size, self.embedding_dim],
                                          initializer=tf.random_uniform_initializer())
            embedded_X = tf.nn.embedding_lookup(embedding_W, inputs)
            embedded_X = tf.cast(embedded_X, tf.float32)
            
        return embedded_X
    
    
    def encoder(self, name, inputs, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size)

            (f_outputs, b_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                        cell_bw, 
                                                                        inputs=inputs, 
                                                                        dtype=tf.float32)

            
            annotation = tf.concat([f_outputs, b_outputs], axis=2)
            
        return annotation
    
    
    def attention(self, name, inputs, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            W_w = tf.get_variable(shape=[self.hidden_size * 2, self.hidden_size], name='weights', initializer=self.initializer)
            b_w = tf.get_variable(shape=[self.hidden_size], name='bias', initializer=self.initializer)
            cv = tf.get_variable(shape=[self.hidden_size], name='context_vector', initializer=self.initializer)
            
            u = tf.tensordot(inputs, W_w, axes=1)
            u = tf.tanh(tf.tensordot(inputs, W_w, axes=1) + b_w)
            u = tf.tensordot(u, cv, axes=1)
            alpha = tf.nn.softmax(u, name='importance_weights')
            
            if name == 'word_attention':
                output = tf.reduce_sum(tf.multiply(inputs, tf.reshape(alpha, [-1, self.sent_num, self.word_num, 1])), 
                                       2, 
                                       name="sentence_vector")
            elif name == 'sent_attention':
                output = tf.reduce_sum(tf.multiply(inputs, tf.reshape(alpha, [-1, self.sent_num, 1])), 
                                       1, 
                                       name="document_vector")
        
        return output
    
    
    def classification(self, inputs, reuse=False):
        with tf.variable_scope("classification", reuse=reuse):
            W_c = tf.get_variable(shape=[self.hidden_size * 2, self.num_classes], name="c_weights", initializer=self.initializer)
            b_c = tf.get_variable(shape=[self.num_classes], name="c_bias", initializer=tf.constant_initializer(0.0))
            
            y_hat = tf.nn.softmax(tf.nn.xw_plus_b(inputs, W_c, b_c))
            
        return y_hat
            
    
    def build_graph(self, inputs, reuse, pretrained=None):
        if self.use_pretrained:
            embedded_X = tf.nn.embedding_lookup(pretrained, inputs)
            embedded_X = tf.cast(embedded_X, tf.float32)
        else:
            embedded_X = self.word_embedding(inputs, reuse=reuse)
            
        embedded_X = tf.reshape(embedded_X, [-1, self.word_num, self.embedding_dim])
            
        word_annotation = self.encoder('word_encoder', embedded_X, reuse=reuse)
        word_annotation = tf.reshape(word_annotation, 
                                     [-1 , self.sent_num, self.word_num, self.hidden_size * 2])
        sent_vectors = self.attention('word_attention', word_annotation, reuse=reuse)
        
        sent_annotation = self.encoder('sent_encoder', sent_vectors, reuse=reuse)
        doc_vectors = self.attention('sent_attention', sent_annotation, reuse=reuse)
        
        y_hat = self.classification(doc_vectors, reuse=reuse)
            
        return y_hat 

