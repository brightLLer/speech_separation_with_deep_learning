#!/home/brightLLer/anaconda3/envs/speech/bin/python
import keras
import librosa
import numpy as np
import os, time
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import mir_eval
from context import *


# os.mkdir('/home/brightLLer/jupyter_notebook/speech/180403/haisen/')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction = 0.43

def write_summary(tensor):
    mean = tf.reduce_mean(tensor)
    std = tf.sqrt(tf.reduce_sum(tf.square(tensor - mean)))
    max_val = tf.reduce_max(tensor)
    min_val = tf.reduce_min(tensor)
    tf.summary.scalar('{}_mean'.format(tensor.op.name), mean)
    tf.summary.scalar('{}_std'.format(tensor.op.name), std)              
    tf.summary.scalar('{}_max'.format(tensor.op.name), max_val)
    tf.summary.scalar('{}_min'.format(tensor.op.name), min_val)
    tf.summary.histogram('{}_histogram'.format(tensor.op.name), tensor)

def create_placeholder():
    X = tf.placeholder(tf.complex64, [None, time_steps, n_features])
    Y1 = tf.placeholder(tf.complex64, [None, time_steps, n_features])
    Y2 = tf.placeholder(tf.complex64, [None, time_steps, n_features])
    return X, Y1, Y2

# build one basic rnn cell
def basic_cell(num_units, activation):
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units, activation=tf.nn.relu)
    return basic_cell
    
def inference(X, num_units=150, num_rnn_layers=2, activation=tf.nn.relu):
    X_amp = tf.abs(X)
    # combined two basic rnn cells with one multilayer cell as we expect two rnn layers
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell(num_units, activation) for _ in range(num_rnn_layers)])
    
    # use the combined cell two build a rnn,get the outputs of the last rnn layer and the last states of two rnn layers
    hidden_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X_amp, dtype=tf.float32)
    
    # stack last rnn layer's outputs to feed to fully connected layer
    stacked_rnn_outputs = tf.reshape(hidden_outputs, [-1, num_units], name='rnn_outputs')
    write_summary(stacked_rnn_outputs)
    
    stacked_Y = tf.contrib.layers.fully_connected(stacked_rnn_outputs, n_features * 2, \
                                                  activation_fn=None)
    write_summary(stacked_Y)
    Y = tf.reshape(stacked_Y, [-1, time_steps, n_features * 2])
    
    # build the mask for the source1 and source2
    Y1_ = Y[:, :, :n_features]
    Y2_ = Y[:, :, n_features:]
    M1 = tf.abs(Y1_) / (tf.abs(Y1_) + tf.abs(Y2_))
    M2 = tf.abs(Y2_) / (tf.abs(Y1_) + tf.abs(Y2_))
    
    # estimate the source1 and source2
    Y_pred1 = tf.multiply(X, tf.cast(M1, tf.complex64), name='source1')
    Y_pred2 = tf.multiply(X, tf.cast(M2, tf.complex64), name='source2')
    
    for parameter in tf.trainable_variables():
        write_summary(parameter)
    return Y_pred1, Y_pred2

def compute_cost(Y_true1, Y_true2, Y_pred1, Y_pred2, gamma=0.0):
    Y_true_amp1 = tf.abs(Y_true1)
    Y_true_amp2 = tf.abs(Y_true2)
    Y_pred_amp1 = tf.abs(Y_pred1)
    Y_pred_amp2 = tf.abs(Y_pred2)
    cost = tf.reduce_sum(tf.square(Y_pred_amp1 - Y_true_amp1)) \
            - gamma * tf.reduce_sum(tf.square(Y_pred_amp1 - Y_true_amp2)) \
            + tf.reduce_sum(tf.square(Y_pred_amp2 - Y_true_amp2)) \
            - gamma * tf.reduce_sum(tf.square(Y_pred_amp2 - Y_true_amp1))
    cost_summary = tf.summary.scalar('cost', cost)
    return cost, cost_summary

def fit_model(start_learning_rate=3e-4, n_epochs=n_epochs, batch_size=batch_size, \
              n_samples=n_samples, val_set=None, bss_eval_set=None):
    
    # logdir
    root_logdir = os.path.join(root, 'logdir/')
    curtime = time.strftime('%Y%m%d%H%M%s')
    logdir = os.path.join(root_logdir, 'run-{}'.format(curtime))
    
    # construction phase
    # clear all the other ops in the default graph
    tf.reset_default_graph()
    
    # create placeholder feeded via mixtures, source#1 and source#2 
    X, Y_true1, Y_true2 = create_placeholder()
    
    # forward propagation
    Y_pred1, Y_pred2 = inference(X)
    
    # compute the cost between ground truth and predictive value
    cost, cost_summary = compute_cost(Y_true1, Y_true2, Y_pred1, Y_pred2, gamma=0.1)
    
    # use adam optimizer to minmize the loss
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    # one train step
    train_step = optimizer.minimize(cost, global_step=global_step)
    
    # init all the global variables
    init = tf.global_variables_initializer()
    
    # merge all summary ops
    # summaries = [sum_op for sum_op in tf.get_collection(tf.GraphKeys.SUMMARIES)]
    summaries = tf.summary.merge_all()
    
    # use the filewriter to write the graph definition and the training stats
    # remember that the train stats and the test stats must use different directory, or the test_writer will overwrite
    # the train_writer
    train_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'train'), graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'test'), graph=tf.get_default_graph())
    
    # use the save node
    saver = tf.train.Saver()
    
    # execution phase
    with tf.Session(config=config) as sess:
        
        # use the filewriter to write the graph definition and the training stats
        # train_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        # test_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        
        # run init
        sess.run(init)
        
        # loop and training
        for epoch in range(n_epochs):
            train_cost = 0
            for k in range(n_batch):
                mix_batch, male_batch, female_batch = flow_from_directory(k * batch_size, (k + 1) * batch_size)
                _, cost_per_batch = sess.run([train_step, cost], \
                                             feed_dict={X: mix_batch, Y_true1: male_batch, Y_true2: female_batch})
                train_cost += cost_per_batch / n_samples
                if (k + 1) % 5 == 0:
                    print('{} batch/{} epoch, the cost of the batch: {:.3f}'.format(k + 1, epoch + 1, cost_per_batch))
    
            if n_samples % batch_size != 0:
                mix_batch, male_batch, female_batch = flow_from_directory(k * batch_size, n_samples)
                _, cost_per_batch = sess.run([train_step, cost], \
                                             feed_dict={X: mix_batch, Y_true1: male_batch, Y_true2: female_batch})
                train_cost += cost_per_batch / n_samples
                
            summary_str = sess.run(summaries, \
                                           feed_dict={X: mix_batch, Y_true1: male_batch, Y_true2: female_batch})
            # print(summary_str)
            train_writer.add_summary(summary_str, epoch)
            # don't forget to flush the stats
            train_writer.flush()
                    
            print('cost after {} epoch: {:.3f}'.format(epoch + 1, train_cost))
            # validation
            if val_set:
                (X_val, Y1_val, Y2_val) = val_set
                test_cost = sess.run(cost, feed_dict={X: X_val, Y_true1: Y1_val, Y_true2: Y2_val})
                print('val cost after {} epoch: {:.3f}'.format(epoch + 1, test_cost))                
                test_cost_summary_str = sess.run(cost_summary, feed_dict={X: X_val, Y_true1: Y1_val, Y_true2: Y2_val})
                test_writer.add_summary(test_cost_summary_str, epoch)
                # don't forget to flush the stats
                test_writer.flush()
            if bss_eval_set:
                (x_val, y1_val, y2_val) = bss_eval_set
                Y1_estimated, Y2_estimated = sess.run([Y_pred1, Y_pred2], feed_dict={X: X_val})
                Y1_estimated = np.squeeze(Y1_estimated).T
                Y2_estimated = np.squeeze(Y2_estimated).T
                y1_estimated = librosa.istft(Y1_estimated, hop_length=512, win_length=1024, length=53040)
                y2_estimated = librosa.istft(Y2_estimated, hop_length=512, win_length=1024, length=53040)
                estimated_sources = np.concatenate([y1_estimated[np.newaxis, :], y2_estimated[np.newaxis, :]], axis=0)
                reference_sources = np.concatenate([y1_val[np.newaxis, :], y2_val[np.newaxis, :]], axis=0)
                sdr, sir, sar, _ =  \
                    mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, compute_permutation=False)
                print('val SDR is {:.3f}, SIR is {:.3f}, SAR is {:.3f}'.format(np.mean(sdr), np.mean(sir), np.mean(sar)))
            
            if (epoch + 1) % 2 == 0:
                saver.save(sess, os.path.join(root, 'weights/model_{:.6f}_{:d}_{:.3f}.ckpt'.format(start_learning_rate, batch_size, test_cost)), global_step=epoch)
        # do not forget to close the writer
        train_writer.close()
        test_writer.close()
            
# X_val, Y1_val, Y2_val, x_val, y1_val, y2_val = load_val_data()
# fit_model(val_set=(X_val, Y1_val, Y2_val), bss_eval_set=(x_val, y1_val, y2_val))