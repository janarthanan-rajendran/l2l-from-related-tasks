from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, vectorize_candidates_sparse, tokenize, r_load_dialog_task
from sklearn import metrics
from memn2n import MemN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import numpy as np
import os
import pickle
import random
import time

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("aux_learning_rate", 0.001, "Learning rate for aux Optimizer that updates anet using aux loss.")
tf.flags.DEFINE_float("outer_learning_rate", 0.001, "Learning rate for qnet Optimizer that updates qnet")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "task id, 1 <= id <= 5")
tf.flags.DEFINE_integer("r_task_id", 5, "task id of the related task, 1 <= id <= 5")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "../data/personalized-dialog-dataset/full", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("test_data_dir", "../data/personalized-dialog-dataset/full", "Directory testing tasks")
tf.flags.DEFINE_string("r_data_dir", "../data/dialog-bAbI-tasks", "Directory containing original bAbI tasks")
tf.flags.DEFINE_string("model_dir", "gen/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_string("aux_opt", "adam", "optimizer for updating anet using aux loss")
tf.flags.DEFINE_boolean('has_qnet', False, 'if True, add question network')
tf.flags.DEFINE_boolean('train', True, 'if True, begin to train')
tf.flags.DEFINE_boolean('sep_test', False, 'if True, load test data from a test data dir')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
tf.flags.DEFINE_boolean('save_vocab', False, 'if True, saves vocabulary')
tf.flags.DEFINE_boolean('load_vocab', False, 'if True, loads vocabulary instead of building it')
tf.flags.DEFINE_boolean('alternate', True, 'if True, alternate training between primary and related every epoch, else do it every batch')
tf.flags.DEFINE_boolean('only_aux', False, 'if True, train anet using only aux, update qnet using full primary task data')


FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)


class chatBot(object):
    def __init__(self, data_dir, r_data_dir, model_dir, result_dir, task_id, r_task_id,
                 OOV=False,
                 has_qnet =False,
                 memory_size=250,
                 random_state=None,
                 batch_size=32,
                 learning_rate=0.001,
                 epsilon=1e-8,
                 max_grad_norm=40.0,
                 evaluation_interval=10,
                 hops=3,
                 epochs=200,
                 embedding_size=20,
                 save_vocab=False,
                 load_vocab=False,
                 alternate=True,
                 only_aux=False,
                 aux_opt='adam',
                 aux_learning_rate=0.001,
                 outer_learning_rate=0.001):
        """Creates wrapper for training and testing a chatbot model.

        Args:
            data_dir: Directory containing personalized dialog tasks.

            r_data_dir: Directory containing related task's data
            
            model_dir: Directory containing memn2n model checkpoints.

            aux_opt: Optimizer for updating anet using aux loss.

            task_id: Personalized dialog task id, 1 <= id <= 5. Defaults to `1`.

            r_task_id: Related tasks task id.

            OOV: If `True`, use OOV test set. Defaults to `False`

            has_qnet: If True, add question network

            memory_size: The max size of the memory. Defaults to `250`.

            random_state: Random state to set graph-level random seed. Defaults to `None`.

            batch_size: Size of the batch for training. Defaults to `32`.

            learning_rate: Learning rate for Adam Optimizer. Defaults to `0.001`.

            epsilon: Epsilon value for Adam Optimizer. Defaults to `1e-8`.

            max_gradient_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            evaluation_interval: Evaluate and print results every x epochs. 
            Defaults to `10`.

            hops: The number of hops over memory for responding. A hop consists 
            of reading and addressing a memory slot. Defaults to `3`.

            epochs: Number of training epochs. Defualts to `200`.

            embedding_size: The size of the word embedding. Defaults to `20`.

            save_vocab: If `True`, save vocabulary file. Defaults to `False`.

            load_vocab: If `True`, load vocabulary from file. Defaults to `False`.

            alternate: If True alternate between primary and related every epoch

            only_aux: Update anet using only aux and update qnet

            aux_learning_rate: lr of aux update to anet

            outer_learning_rate: lr for update qnet
        """

        self.data_dir = data_dir
        self.r_data_dir = r_data_dir
        self.task_id = task_id
        self.r_task_id = r_task_id
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.OOV = OOV
        self.has_qnet = has_qnet
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.save_vocab = save_vocab
        self.load_vocab = load_vocab
        self.alternate = alternate
        self.only_aux = only_aux
        self.aux_opt = aux_opt
        self.aux_learning_rate = aux_learning_rate
        self.outer_learning_rate = outer_learning_rate

        candidates,self.candid2indx = load_candidates(self.data_dir, self.task_id, True)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key],key) 
                                for key in self.candid2indx)

        if self.has_qnet:
            r_candidates, self.r_candid2indx = load_candidates(self.r_data_dir, self.r_task_id, False)
            self.r_n_cand = len(r_candidates)
            print("R Candidate Size", self.r_n_cand)
            self.r_indx2candid = dict((self.r_candid2indx[key], key)
                                    for key in self.r_candid2indx)
        
        # Task data
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, self.OOV)
        data = self.trainData + self.testData + self.valData

        if self.has_qnet:
            self.r_trainData, _, _ = r_load_dialog_task(
                self.r_data_dir, self.r_task_id, self.r_candid2indx, self.OOV)
            data = data + self.r_trainData
        
        self.build_vocab(data,candidates,self.save_vocab,self.load_vocab)
        
        self.candidates_vec = vectorize_candidates(
            candidates,self.word_idx,self.candidate_sentence_size)

        if FLAGS.sep_test:
            _, self.sep_testData, _ = load_dialog_task(
                FLAGS.test_data_dir, self.task_id, self.candid2indx, self.OOV)
        
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)

        if self.aux_opt == 'sgd':
            aux_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.aux_learning_rate)
        elif self.aux_opt == 'adam':
            aux_optimizer = tf.train.AdamOptimizer(learning_rate=self.aux_learning_rate, epsilon=self.epsilon)

        outer_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.outer_learning_rate, epsilon=self.epsilon)

        # config.gpu_options.per_process_gpu_memory_fraction = 0.5

        self.sess = tf.Session(config=config)

        self.model = MemN2NDialog(self.has_qnet, self.batch_size, self.vocab_size, self.n_cand,
                                  self.sentence_size, self.embedding_size,
                                  self.candidates_vec, self.candidate_sentence_size, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm,
                                  optimizer=optimizer, outer_optimizer= outer_optimizer, aux_optimizer=aux_optimizer, task_id=task_id,
                                  inner_lr = self.aux_learning_rate)

        self.saver = tf.train.Saver(max_to_keep=50)
        
        self.summary_writer = tf.summary.FileWriter(
            self.result_dir, self.model.graph_output.graph)
        
    def build_vocab(self,data,candidates,save=False,load=False):
        """Build vocabulary of words from all dialog data and candidates."""
        if load:
            # Load from vocabulary file
            vocab_file = open('vocab.obj', 'rb')
            vocab = pickle.load(vocab_file)
        else:
            if self.has_qnet:
                vocab = reduce(lambda x, y: x | y,
                               (set(list(chain.from_iterable(s)) + q + q_a)
                                 for s, q, a, q_a in data))
            else:
                vocab = reduce(lambda x, y: x | y,
                               (set(list(chain.from_iterable(s)) + q)
                                for s, q, a, q_a in data))

            vocab |= reduce(lambda x,y: x|y, 
                            (set(candidate) for candidate in candidates) )
            vocab = sorted(vocab)
        
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (s for s, _, _, _ in data)))
        mean_story_size = int(np.mean([ len(s) for s, _, _, _ in data ]))
        self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _ in data)))
        self.candidate_sentence_size=max(map(len,candidates))
        query_size = max(map(len, (q for _, q, _, _ in data)))
        q_answer_size = max(map(len, (q_a for _, _, _, q_a in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        if self.has_qnet:
            self.sentence_size = max(query_size, self.sentence_size, q_answer_size)  # for the position
        else:
            self.sentence_size = max(query_size, self.sentence_size)  # for the position

        # Print parameters
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length", self.candidate_sentence_size)
        print("Longest story length", max_story_size)
        print("Average story length", mean_story_size)

        # Save to vocabulary file
        if save:
            vocab_file = open('vocab.obj', 'wb')
            pickle.dump(vocab, vocab_file)    
        
    def train(self):
        """Runs the training algorithm over training set data.

        Performs validation at given evaluation intervals.
        """
        trainS, trainQ, trainA, trainqA = vectorize_data(
            self.trainData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
            self.batch_size, self.n_cand, self.memory_size)
        if self.has_qnet:
            r_trainS, r_trainQ, r_trainA, r_trainqA = vectorize_data(
                self.r_trainData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
                self.batch_size, self.r_n_cand, self.memory_size)
            n_r_train = len(r_trainS)

        valS, valQ, valA, _ = vectorize_data(
            self.valData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
            self.batch_size, self.n_cand, self.memory_size)
        n_train = len(trainS)
        n_val = len(valS)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train-self.batch_size, self.batch_size), 
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy=0

        if self.has_qnet:
            np.random.shuffle(batches)
            p_batches = batches[:int(len(batches)/2)]
            r_batches_p = batches[int(len(batches)/2):]
            # p_batches = zip(range(0, int(n_train/2) - self.batch_size, self.batch_size),
            #               range(self.batch_size, int(n_train/2), self.batch_size))
            # p_batches = [(start, end) for start, end in p_batches]
            # # primary data for the related tasks training (qnet training)
            # r_batches_p = zip(range(int(n_train/2), n_train - self.batch_size, self.batch_size),
            #               range(int(n_train/2) + self.batch_size, n_train, self.batch_size))
            # r_batches_p = [(start, end) for start, end in r_batches_p]
            r_batches_r = zip(range(0, n_r_train-self.batch_size, self.batch_size),
                          range(self.batch_size, n_r_train, self.batch_size))
            r_batches_r = [(start, end) for start, end in r_batches_r]

        # Training loop
        start_time = time.process_time()
        for t in range(1, self.epochs+1):
            print('Epoch', t)
            np.random.shuffle(batches)
            total_cost = 0.0
            if self.has_qnet:
                np.random.shuffle(p_batches)
                np.random.shuffle(r_batches_p)
                np.random.shuffle(r_batches_r)

                if self.only_aux:
                    for r_start, r_end in r_batches_r:
                        start, end = random.sample(batches, 1)[0]
                        r_s_p = trainS[start:end]
                        r_q_p = trainQ[start:end]
                        r_a_p = trainA[start:end]
                        r_q_a_p = trainqA[start:end]
                        r_s = r_trainS[r_start:r_end]
                        r_q = r_trainQ[r_start:r_end]
                        r_a = r_trainA[r_start:r_end]
                        r_q_a = r_trainqA[r_start:r_end]
                        # print('s', np.shape(s), 'q', np.shape(q), 'a', np.shape(a), 'q_a', np.shape(q_a))
                        outer_cost_t, aux_cost_t = self.model.q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p,
                                                                          False)  # related
                        # outer_cost_t, aux_cost_t = self.model.q_batch_fit(s, q, a, q_a, s, q, a, False)  # related
                        cost_t = outer_cost_t
                        # print('outer_cost', outer_cost_t, 'aux_cost', aux_cost_t)
                        total_cost += cost_t
                else:
                    if self.alternate:
                        if t % 2 == 0:
                            for start, end in p_batches:
                                s = trainS[start:end]
                                q = trainQ[start:end]
                                a = trainA[start:end]
                                q_a = trainqA[start:end]
                                cost_t = self.model.q_batch_fit(s, q, a, q_a, None, None, None, True)  # primary
                                # print('primary cost', cost_t)
                                total_cost += cost_t
                        else:
                            for r_start,r_end in r_batches_r:
                                start, end = random.sample(r_batches_p,1)[0]
                                r_s_p = trainS[start:end]
                                r_q_p = trainQ[start:end]
                                r_a_p = trainA[start:end]
                                r_q_a_p = trainqA[start:end]
                                r_s = r_trainS[r_start:r_end]
                                r_q = r_trainQ[r_start:r_end]
                                r_a = r_trainA[r_start:r_end]
                                r_q_a = r_trainqA[r_start:r_end]
                                # print('s', np.shape(s), 'q', np.shape(q), 'a', np.shape(a), 'q_a', np.shape(q_a))
                                outer_cost_t, aux_cost_t = self.model.q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p, False)  # related
                                # outer_cost_t, aux_cost_t = self.model.q_batch_fit(s, q, a, q_a, s, q, a, False)  # related
                                cost_t = outer_cost_t
                                # print('outer_cost', outer_cost_t, 'aux_cost', aux_cost_t)
                                total_cost += cost_t
                    else:
                        for start, end in p_batches:
                            s = trainS[start:end]
                            q = trainQ[start:end]
                            a = trainA[start:end]
                            q_a = trainqA[start:end]
                            cost_t = self.model.q_batch_fit(s, q, a, q_a, None, None, None, True)  # primary
                            # print('primary cost', cost_t)
                            total_cost += cost_t

                            r_start, r_end = random.sample(r_batches_r, 1)[0]
                            r_start_p, r_end_p = random.sample(r_batches_p, 1)[0]
                            r_s_p = trainS[r_start_p:r_end_p]
                            r_q_p = trainQ[r_start_p:r_end_p]
                            r_a_p = trainA[r_start_p:r_end_p]
                            r_q_a_p = trainqA[r_start_p:r_end_p]
                            r_s = r_trainS[r_start:r_end]
                            r_q = r_trainQ[r_start:r_end]
                            r_a = r_trainA[r_start:r_end]
                            r_q_a = r_trainqA[r_start:r_end]
                            # print('s', np.shape(s), 'q', np.shape(q), 'a', np.shape(a), 'q_a', np.shape(q_a))
                            outer_cost_t, aux_cost_t = self.model.q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p,
                                                                              False)  # related
                            # outer_cost_t, aux_cost_t = self.model.q_batch_fit(s, q, a, q_a, s, q, a, False)  # related
                            cost_t = outer_cost_t
                            # print('outer_cost', outer_cost_t, 'aux_cost', aux_cost_t)
                            total_cost += cost_t
            else:
                for start, end in batches:
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    a = trainA[start:end]
                    # q_a = trainqA[start:end]
                    cost_t = self.model.batch_fit(s, q, a)
                    total_cost += cost_t
            if t % self.evaluation_interval == 0:
                # Perform validation
                if self.has_qnet and not self.only_aux:
                    train_preds = self.batch_predict(trainS[:int(n_train/2)],trainQ[:int(n_train/2)],int(n_train/2))
                    train_acc = metrics.accuracy_score(np.array(train_preds), trainA[:int(n_train/2)])
                else:
                    train_preds = self.batch_predict(trainS,trainQ,n_train)
                    train_acc = metrics.accuracy_score(np.array(train_preds), trainA)
                val_preds = self.batch_predict(valS,valQ,n_val)
                val_acc = metrics.accuracy_score(val_preds, valA)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')

                # Write summary
                train_acc_summary = tf.summary.scalar(
                    'task_' + str(self.task_id) + '/' + 'train_acc', 
                    tf.constant((train_acc), dtype=tf.float32))
                val_acc_summary = tf.summary.scalar(
                    'task_' + str(self.task_id) + '/' + 'val_acc', 
                    tf.constant((val_acc), dtype=tf.float32))
                merged_summary = tf.summary.merge([train_acc_summary, val_acc_summary])
                summary_str = self.sess.run(merged_summary)
                self.summary_writer.add_summary(summary_str, t)
                self.summary_writer.flush()
                
                if val_acc > best_validation_accuracy:
                    best_validation_accuracy=val_acc
                    self.saver.save(self.sess,self.model_dir+'model.ckpt',
                                    global_step=t)
        time_taken = time.process_time() - start_time
        print("Time taken", time_taken)


    def test(self):
        """Runs testing on testing set data.

        Loads best performing model weights based on validation accuracy.
        """
        model_dir = 'model/' + str(FLAGS.task_id) + '/' + FLAGS.model_dir
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")

        if FLAGS.sep_test:
            testS, testQ, testA, _ = vectorize_data(
                self.sep_testData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
                self.batch_size, self.n_cand, self.memory_size)
        else:
            testS, testQ, testA, _ = vectorize_data(
                self.testData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
                self.batch_size, self.n_cand, self.memory_size)
        n_test = len(testS)
        print("Testing Size", n_test)
        
        test_preds = self.batch_predict(testS, testQ, n_test)
        test_acc = metrics.accuracy_score(test_preds, testA)
        print("Testing Accuracy:", test_acc)
        
        # # Un-comment below to view correct responses and predictions 
        # print(testA)
        # for pred in test_preds:
        #    print(pred, self.indx2candid[pred])

    def batch_predict(self,S,Q,n):
        """Predict answers over the passed data in batches.

        Args:
            S: Tensor (None, memory_size, sentence_size)
            Q: Tensor (None, sentence_size)
            n: int

        Returns:
            preds: Tensor (None, vocab_size)
        """
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            pred = self.model.predict(s, q)
            preds += list(pred)
        return preds

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':

    # # config = tf.ConfigProto()
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #    try:
    #        # Currently, memory growth needs to be the same across GPUs
    #        for gpu in gpus:
    #            tf.config.experimental.set_memory_growth(gpu, True)
    #        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #    except RuntimeError as e:
    #        # Memory growth must be set before GPUs have been initialized
    #        print(e)

    # tf.config.experimental.set_memory_growth(gpu, True)

    model_dir = 'model/' + str(FLAGS.task_id) + '/' + FLAGS.model_dir
    result_dir = 'result/' + str(FLAGS.task_id) + '/' + FLAGS.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    chatbot = chatBot(FLAGS.data_dir, FLAGS.r_data_dir, model_dir, result_dir, FLAGS.task_id, FLAGS.r_task_id, OOV=FLAGS.OOV,
                      has_qnet=FLAGS.has_qnet, batch_size=FLAGS.batch_size, memory_size=FLAGS.memory_size,
                      epochs=FLAGS.epochs, hops=FLAGS.hops, save_vocab=FLAGS.save_vocab,
                      load_vocab=FLAGS.load_vocab, learning_rate=FLAGS.learning_rate,
                      embedding_size=FLAGS.embedding_size, evaluation_interval=FLAGS.evaluation_interval,
                      alternate=FLAGS.alternate, only_aux=FLAGS.only_aux, aux_opt=FLAGS.aux_opt,
                      aux_learning_rate=FLAGS.aux_learning_rate, outer_learning_rate=FLAGS.outer_learning_rate)

    if FLAGS.train:
        chatbot.train()
    else:
        chatbot.test()
    
    chatbot.close_session()
