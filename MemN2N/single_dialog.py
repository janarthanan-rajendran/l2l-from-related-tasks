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
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam, rms Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 0.5, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 5, "task id, 1 <= id <= 5")
tf.flags.DEFINE_integer("r_task_id", 5, "task id of the related task, 1 <= id <= 5")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "../data/personalized-dialog-dataset/full", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("test_data_dir", "../data/personalized-dialog-dataset/full", "Directory testing tasks")
tf.flags.DEFINE_string("r_data_dir", "../data/dialog-bAbI-tasks", "Directory containing original bAbI tasks")
tf.flags.DEFINE_string("model_dir", "gen/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_string("restore_model_dir", "gen/", "Directory containing model for restore for training")
tf.flags.DEFINE_boolean('restore', False, 'if True,restore for training')
tf.flags.DEFINE_string("aux_opt", "adam", "optimizer for updating anet using aux loss")
tf.flags.DEFINE_string("aux_nonlin", None, "nonlinearity at the end of aux prediction/target, arctan")
tf.flags.DEFINE_boolean('has_qnet', True, 'if True, add question network')
tf.flags.DEFINE_boolean('train', True, 'if True, begin to train')
tf.flags.DEFINE_boolean('sep_test', False, 'if True, load test data from a test data dir')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
tf.flags.DEFINE_boolean('save_vocab', False, 'if True, saves vocabulary')
tf.flags.DEFINE_boolean('load_vocab', False, 'if True, loads vocabulary instead of building it')
tf.flags.DEFINE_boolean('alternate', True, 'if True, alternate training between primary and related every epoch, else do it every batch')
tf.flags.DEFINE_boolean('only_aux', False, 'if True, train anet using only aux, update qnet using full primary task data')
tf.flags.DEFINE_boolean('only_primary', False, 'if True, train anet using only primary')
tf.flags.DEFINE_boolean('m_series', False, 'if True, m_series is set')
tf.flags.DEFINE_boolean('only_related', False, 'if True, train qnet using only related tasks')
tf.flags.DEFINE_boolean('copy_qnet2anet', False, 'if True copy qnet to anet before starting training')
tf.flags.DEFINE_boolean('transform_qnet', False, 'if True train qnet_aux with primary data to match anet u_k')
tf.flags.DEFINE_boolean('transform_anet', False, 'if True train anet(u_k) with related data to match qnet_aux')
tf.flags.DEFINE_boolean('primary_and_related', False, 'if True train anet(u_k) with related data  and primary data')
tf.flags.DEFINE_boolean('gated_qnet', False, 'gated qnet')
tf.flags.DEFINE_float("outer_r_weight", 0, "Weight of the related task loss in the outer loop")
tf.flags.DEFINE_integer("qnet_hops", 3, "Number of hops in the qnet Memory Network.")
tf.flags.DEFINE_boolean('copy_qnet2gqnet', False, 'if True copy qnet to gated qnet before starting training')
tf.flags.DEFINE_boolean('separate_eval', False, 'if True split eval data from primary')
tf.flags.DEFINE_boolean('r1', False, 'if True second related task')
tf.flags.DEFINE_string("r1_data_dir", "../data/personalized-dialog-dataset/small-r1-10", "Directory containing r1 related tasks")
tf.flags.DEFINE_string("gate_nonlin", None, "nonlinearity at the end gated qnet")
tf.flags.DEFINE_boolean('only_gated_qnet', False, 'if True update only gated qnet')
tf.flags.DEFINE_boolean('only_gated_aux', False, 'if True update only anet with gated_aux')
tf.flags.DEFINE_boolean('only_gated_aux_primary', False, 'if True update only anet with gated aux and with primary')
tf.flags.DEFINE_integer("inner_steps", 1, "Number of inner loop steps")




FLAGS = tf.flags.FLAGS
print("Started Task :)) :", FLAGS.task_id)


class chatBot(object):
    def __init__(self, data_dir, r_data_dir, model_dir, result_dir, task_id, r_task_id,
                 OOV=False,
                 has_qnet =False,
                 memory_size=250,
                 random_state=None,
                 batch_size=32,
                 learning_rate=0.001,
                 epsilon=1e-8,
                 alpha=0.9,
                 max_grad_norm=0.5,
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
                 outer_learning_rate=0.001,
                 only_primary=False,
                 aux_nonlin=None,
                 m_series=False,
                 only_related=False,
                 transform_qnet=False,
                 transform_anet=False,
                 primary_and_related=False,
                 gated_qnet=False,
                 outer_r_weight=0):
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

            alpha: decay of rmsprop optimizer.

            max_gradient_norm: Maximum L2 norm clipping value. Defaults to `0.5`.

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

            only_primary: train on only primary data

            aux_nonlin: non linearity at the end of aux pred/targ

            m_series: m_series is set if true

            only_related: If true train qnet with related task data
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
        self.alpha = alpha
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
        self.only_primary = only_primary
        self.aux_nonlin = aux_nonlin
        self.m_series = m_series
        self.only_related = only_related
        self.transform_qnet = transform_qnet
        self.transform_anet = transform_anet
        self.primary_and_related = primary_and_related
        self.gated_qnet = gated_qnet
        self.outer_r_weight = outer_r_weight

        candidates,self.candid2indx = load_candidates(self.data_dir, self.task_id, True, FLAGS.r1)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key],key) 
                                for key in self.candid2indx)

        if self.has_qnet:
            r_candidates, self.r_candid2indx = load_candidates(self.r_data_dir, self.r_task_id, False, FLAGS.r1)
            self.r_n_cand = len(r_candidates)
            print("R Candidate Size", self.r_n_cand)
            self.r_indx2candid = dict((self.r_candid2indx[key], key)
                                    for key in self.r_candid2indx)
        
        # Task data
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, self.OOV)
        data = self.trainData + self.testData + self.valData

        if self.has_qnet:
            self.r_trainData, self.r_testData, self.r_valData = r_load_dialog_task(
                self.r_data_dir, self.r_task_id, self.r_candid2indx, self.OOV)
            data = data + self.r_trainData + self.r_valData + self.r_testData

            if FLAGS.r1:
                self.r1_trainData, _, _ = load_dialog_task(
                    FLAGS.r1_data_dir, self.task_id, self.r_candid2indx, self.OOV)
                data = data + self.r1_trainData

        if self.has_qnet:
            self.build_vocab(data,candidates,self.save_vocab,self.load_vocab, r_candidates)
        
        self.candidates_vec = vectorize_candidates(
            candidates,self.word_idx,self.candidate_sentence_size)

        if self.has_qnet:
            self.r_candidates_vec = vectorize_candidates(
                r_candidates,self.word_idx,self.r_candidate_sentence_size)
        else:
            self.r_candidates_vec = None

        if FLAGS.sep_test:
            _, self.sep_testData, _ = load_dialog_task(
                FLAGS.test_data_dir, self.task_id, self.candid2indx, self.OOV)
        
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon, name='opt')

        if self.aux_opt == 'sgd':
            aux_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.aux_learning_rate, name='aux_opt')
        elif self.aux_opt == 'adam':
            aux_optimizer = tf.train.AdamOptimizer(learning_rate=self.aux_learning_rate, epsilon=self.epsilon, name='aux_opt')
        elif self.aux_opt == 'rms':
            aux_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.aux_learning_rate, decay=self.alpha, epsilon=self.epsilon, name='aux_opt')
        else:
            print("unknown aux optimizer")

        outer_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.outer_learning_rate, epsilon=self.epsilon, name='outer_opt')

        self.sess = tf.Session(config=config)

        self.model = MemN2NDialog(self.has_qnet, self.batch_size, self.vocab_size, self.n_cand,
                                  self.sentence_size, self.embedding_size,
                                  self.candidates_vec, self.candidate_sentence_size, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm,
                                  optimizer=optimizer, outer_optimizer=outer_optimizer, aux_optimizer=aux_optimizer, task_id=task_id,
                                  inner_lr=self.aux_learning_rate, aux_opt_name=self.aux_opt, alpha=self.alpha,
                                  epsilon=self.epsilon, aux_nonlin=self.aux_nonlin, m_series=self.m_series,
                                  r_candidates_vec=self.r_candidates_vec, outer_r_weight=self.outer_r_weight,
                                  qnet_hops = FLAGS.qnet_hops, gate_nonlin=FLAGS.gate_nonlin, inner_steps=FLAGS.inner_steps, r_candidates_size=self.r_n_cand)

        self.saver = tf.train.Saver(max_to_keep=50)
        
        self.summary_writer = tf.summary.FileWriter(
            self.result_dir, self.model.graph_output.graph)
        
    def build_vocab(self,data,candidates,save=False,load=False, r_candidates=None):
        """Build vocabulary of words from all dialog data and candidates."""
        if load:
            # Load from vocabulary file
            vocab_file = open('vocab.obj', 'rb')
            vocab = pickle.load(vocab_file)
        else:
            if self.has_qnet and not self.m_series:
                vocab = reduce(lambda x, y: x | y,
                               (set(list(chain.from_iterable(s)) + q + q_a)
                                 for s, q, a, q_a in data))
            else:
                vocab = reduce(lambda x, y: x | y,
                               (set(list(chain.from_iterable(s)) + q)
                                for s, q, a, q_a in data))

            vocab |= reduce(lambda x,y: x|y, 
                            (set(candidate) for candidate in candidates) )

            if self.has_qnet and self.m_series:
                vocab |= reduce(lambda x, y: x | y,
                                (set(r_candidate) for r_candidate in r_candidates))

            vocab = sorted(vocab)
        
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (s for s, _, _, _ in data)))
        mean_story_size = int(np.mean([ len(s) for s, _, _, _ in data ]))
        self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _ in data)))
        self.candidate_sentence_size=max(map(len,candidates))
        if self.has_qnet:
            self.r_candidate_sentence_size=max(map(len,r_candidates))
        query_size = max(map(len, (q for _, q, _, _ in data)))
        q_answer_size = max(map(len, (q_a for _, _, _, q_a in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        if self.has_qnet and not self.m_series:
            self.sentence_size = max(query_size, self.sentence_size, q_answer_size)  # for the position
        else:
            self.sentence_size = max(query_size, self.sentence_size)  # for the position

        # Print parameters
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length", self.candidate_sentence_size)
        if self.has_qnet and self.m_series:
            print("Longest r_candidate sentence length", self.r_candidate_sentence_size)
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
        if FLAGS.restore:
            model_dir = 'model/' + str(FLAGS.task_id) + '/' + FLAGS.restore_model_dir
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Restored checkpoint")
            else:
                print("...no checkpoint found...")

        trainS, trainQ, trainA, trainqA = vectorize_data(
            self.trainData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
            self.batch_size, self.n_cand, self.memory_size)
        if self.has_qnet:
            r_trainS, r_trainQ, r_trainA, r_trainqA = vectorize_data(
                self.r_trainData, self.word_idx, self.sentence_size, self.r_candidate_sentence_size,
                self.batch_size, self.r_n_cand, self.memory_size)
            n_r_train = len(r_trainS)
            print("Related Task Trainign Size", n_r_train)

            if FLAGS.r1:
                r1_trainS, r1_trainQ, r1_trainA, r1_trainqA = vectorize_data(
                    self.r1_trainData, self.word_idx, self.sentence_size, self.r_candidate_sentence_size,
                    self.batch_size, self.r_n_cand, self.memory_size)
                n_r1_train = len(r1_trainS)
                print("Second Related Task Trainign Size", n_r1_train)
                n_r_orig_train = len(r_trainS[0:int(n_r_train/self.batch_size)*self.batch_size])

                A_qA = list(zip(r1_trainA, r1_trainqA))
                np.random.seed(0)
                np.random.shuffle(A_qA)
                r1_trainA, r1_trainqA = zip(*A_qA)

                # print(type(r_trainA), type(r_trainA[0:int(n_r_train/self.batch_size)*self.batch_size]), type(r1_trainA))

                r_trainS = r_trainS[0:int(n_r_train/self.batch_size)*self.batch_size] + r1_trainS
                r_trainQ = r_trainQ[0:int(n_r_train/self.batch_size)*self.batch_size] + r1_trainQ
                r_trainA = r_trainA[0:int(n_r_train/self.batch_size)*self.batch_size] + list(r1_trainA)
                r_trainqA = r_trainqA[0:int(n_r_train/self.batch_size)*self.batch_size] + list(r1_trainqA)

                n_r_train = len(r_trainS)
                print("joint related task trainin size", n_r_train)

        valS, valQ, valA, _ = vectorize_data(
            self.valData, self.word_idx, self.sentence_size, self.candidate_sentence_size,
            self.batch_size, self.n_cand, self.memory_size)

        if self.has_qnet and self.m_series:
            r_valS, r_valQ, r_valA, _ = vectorize_data(
                self.r_valData, self.word_idx, self.sentence_size, self.r_candidate_sentence_size,
                self.batch_size, self.r_n_cand, self.memory_size)
            n_r_val = len(r_valS)

        n_train = len(trainS)
        n_val = len(valS)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train-self.batch_size, self.batch_size), 
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy=0
        best_validation_loss = np.inf
        best_validation_epoch = 0

        if self.has_qnet:
            np.random.seed(0)
            np.random.shuffle(batches)
            p_batches = batches[:int(len(batches)/2)]
            r_batches_p = batches[int(len(batches)/2):]
            r_batches_r = zip(range(0, n_r_train-self.batch_size, self.batch_size),
                          range(self.batch_size, n_r_train, self.batch_size))
            r_batches_r = [(start, end) for start, end in r_batches_r]


        # Training loop
        start_time = time.process_time()

        if FLAGS.copy_qnet2anet:
            self.model.copy_qnet2anet()
            print("Qnet copied to anet")

        if FLAGS.copy_qnet2gqnet:
            self.model.copy_qnet2gqnet()
            print("Qnet copied to gated qnet")

        for t in range(1, self.epochs+1):
            print('Epoch', t)
            np.random.shuffle(batches)
            total_cost = 0.0
            if self.has_qnet:
                np.random.shuffle(p_batches)
                np.random.shuffle(r_batches_p)
                np.random.shuffle(r_batches_r)
                if self.only_aux:
                    count = 0
                    for r_start, r_end in r_batches_r:
                        count +=1
                        start, end = random.sample(batches, 1)[0]
                        r_s_p = trainS[start:end]
                        r_q_p = trainQ[start:end]
                        r_a_p = trainA[start:end]
                        r_q_a_p = trainqA[start:end]
                        r_s = r_trainS[r_start:r_end]
                        r_q = r_trainQ[r_start:r_end]
                        r_a = r_trainA[r_start:r_end]
                        r_q_a = r_trainqA[r_start:r_end]
                        outer_cost_t, aux_cost_t = self.model.q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p, False)  # related
                        cost_t = outer_cost_t
                        if count%100 == 0:
                            print('outer_cost', outer_cost_t, 'aux_cost', aux_cost_t)
                        total_cost += cost_t
                elif self.only_primary:
                    if FLAGS.separate_eval:
                        for start, end in p_batches:
                            s = trainS[start:end]
                            q = trainQ[start:end]
                            a = trainA[start:end]
                            # q_a = trainqA[start:end]
                            cost_t = self.model.batch_fit(s, q, a)
                            total_cost += cost_t
                    else:
                        for start, end in batches:
                            s = trainS[start:end]
                            q = trainQ[start:end]
                            a = trainA[start:end]
                            # q_a = trainqA[start:end]
                            cost_t = self.model.batch_fit(s, q, a)
                            total_cost += cost_t
                elif self.only_related:
                    for start, end in r_batches_r:
                        s = r_trainS[start:end]
                        q = r_trainQ[start:end]
                        a = r_trainA[start:end]
                        # q_a = trainqA[start:end]
                        cost_t = self.model.q_batch_fit_r(s, q, a)
                        total_cost += cost_t
                elif self.transform_qnet:
                    for start, end in batches:
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        a = trainA[start:end]
                        # q_a = trainqA[start:end]
                        cost_t = self.model.batch_fit_qt(s, q, a)
                        total_cost += cost_t
                elif self.transform_anet:
                    for start, end in r_batches_r:
                        s = r_trainS[start:end]
                        q = r_trainQ[start:end]
                        a = r_trainA[start:end]
                        # q_a = trainqA[start:end]
                        cost_t = self.model.batch_fit_at(s, q, a)
                        total_cost += cost_t
                elif self.primary_and_related:
                    count = 0
                    for r_start, r_end in r_batches_r:
                        s = r_trainS[r_start:r_end]
                        q = r_trainQ[r_start:r_end]
                        a = r_trainA[r_start:r_end]
                        # q_a = trainqA[start:end]
                        cost_t_related = self.model.batch_fit(s, q, a, primary=False)

                        start, end = random.sample(batches, 1)[0]
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        a = trainA[start:end]
                        # q_a = trainqA[start:end]
                        cost_t_primary = self.model.batch_fit(s, q, a)

                        if count % 100 == 0:
                            print("related", cost_t_related, "primary", cost_t_primary)

                        total_cost += cost_t_related + cost_t_primary
                        count += 1
                elif FLAGS.only_gated_qnet:
                    if FLAGS.inner_steps == 1:
                        count = 0
                        gate_r1 = 0
                        gate_joint_r = 0
                        for r_start, r_end in r_batches_r:
                            count += 1
                            if FLAGS.separate_eval:
                                start, end = random.sample(r_batches_p, 1)[0]
                            else:
                                start, end = random.sample(batches, 1)[0]
                            r_s_p = trainS[start:end]
                            r_q_p = trainQ[start:end]
                            r_a_p = trainA[start:end]
                            r_q_a_p = trainqA[start:end]

                            r_s = r_trainS[r_start:r_end]
                            r_q = r_trainQ[r_start:r_end]
                            r_a = r_trainA[r_start:r_end]
                            r_q_a = r_trainqA[r_start:r_end]

                            cost_t_outer, aux_gate = self.model.gated_q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p) #gated qnet update
                            total_cost += cost_t_outer

                            if r_start >= n_r_orig_train:
                                gate_r1 += np.sum(aux_gate)
                            gate_joint_r += np.sum(aux_gate)
                            if count % 100 == 0:
                                print("count", count, "outer", cost_t_outer)
                        print("Ratio of gate_r1/r1: ", gate_r1/n_r1_train, "Ratio of gate_joint_r/joint_r", gate_joint_r/n_r_train)
                    else:
                        count = 0
                        r_s_list = []
                        r_q_list = []
                        r_a_list = []
                        r_q_a_list = []

                        # r_s_list1 = np.zeros((FLAGS.inner_steps, ))

                        for r_start, r_end in r_batches_r:
                            count += 1

                            r_s = r_trainS[r_start:r_end]
                            r_q = r_trainQ[r_start:r_end]
                            r_a = r_trainA[r_start:r_end]
                            r_q_a = r_trainqA[r_start:r_end]

                            r_s_list.append(r_s)
                            # print(np.array(r_s).shape)
                            r_q_list.append(r_q)
                            r_a_list.append(r_a)
                            r_q_a_list.append(r_q_a)

                            if count % FLAGS.inner_steps == 0:
                                if FLAGS.separate_eval:
                                    start, end = random.sample(r_batches_p, 1)[0]
                                else:
                                    start, end = random.sample(batches, 1)[0]
                                r_s_p = trainS[start:end]
                                r_q_p = trainQ[start:end]
                                r_a_p = trainA[start:end]
                                r_q_a_p = trainqA[start:end]

                                #used if outer_r_weight > 0
                                r_start1, r_end1 = random.sample(r_batches_r, 1)[0]
                                r_s1 = r_trainS[r_start1:r_end1]
                                r_q1 = r_trainQ[r_start1:r_end1]
                                r_a1 = r_trainA[r_start1:r_end1]
                                r_q_a1 = r_trainqA[r_start1:r_end1]

                                # print(np.asarray(r_s).shape, np.asarray(r_s_list).shape, np.asarray(r_q_list).shape, np.asarray(r_a_list).shape)
                                cost_t_outer = self.model.gated_q_batch_fit_list(np.asarray(r_s_list), np.asarray(r_q_list),
                                                                                           np.asarray(r_a_list), np.asarray(r_q_a_list),
                                                                                           r_s1, r_q1, r_a1, r_q_a1, r_s_p, r_q_p, r_a_p)  # gated qnet update
                                # print(np.asarray(r_s_list).shape, np.asarray(r_q_list).shape, np.asarray(r_a_list).shape)
                                # cost_t_outer = self.model.gated_q_batch_fit_list(r_s_list, r_q_list,
                                #                                                            r_a_list, r_q_a_list,
                                #                                                            r_s1, r_q1, r_a1, r_q_a1, r_s_p, r_q_p, r_a_p)  # gated qnet update
                                total_cost += cost_t_outer

                                r_s_list = []
                                r_q_list = []
                                r_a_list = []
                                r_q_a_list = []

                            if count % 100 == 0:
                                print("count", count, "outer", cost_t_outer)

                elif FLAGS.only_gated_aux:
                    count = 0
                    gate_r1 = 0
                    gate_joint_r = 0
                    for r_start, r_end in r_batches_r:
                        count += 1
                        r_s = r_trainS[r_start:r_end]
                        r_q = r_trainQ[r_start:r_end]
                        r_a = r_trainA[r_start:r_end]
                        r_q_a = r_trainqA[r_start:r_end]

                        cost_t_aux, aux_gate = self.model.gated_batch_fit(r_s, r_q, r_a) #anet with aux update with related data
                        total_cost += cost_t_aux
                        if r_start >= n_r_orig_train:
                            gate_r1 += np.sum(aux_gate)
                        gate_joint_r += np.sum(aux_gate)
                        if count % 100 == 0:
                            print("count", count, "aux", cost_t_aux)
                            # print("Aux_gate", aux_gate)
                    print("Ratio of gate_r1/r1: ", gate_r1/n_r1_train, "Ratio of gate_joint_r/joint_r", gate_joint_r/n_r_train)

                elif FLAGS.only_gated_aux_primary:
                    count = 0
                    gate_r1 = 0
                    gate_joint_r = 0
                    for r_start, r_end in r_batches_r:
                        count += 1
                        r_s = r_trainS[r_start:r_end]
                        r_q = r_trainQ[r_start:r_end]
                        r_a = r_trainA[r_start:r_end]
                        r_q_a = r_trainqA[r_start:r_end]

                        cost_t_aux, aux_gate = self.model.gated_batch_fit(r_s, r_q, r_a) #anet with aux update with related data
                        if r_start >= n_r_orig_train:
                            gate_r1 += np.sum(aux_gate)
                        gate_joint_r += np.sum(aux_gate)

                        if FLAGS.separate_eval:
                            start, end = random.sample(p_batches, 1)[0]
                        else:
                            start, end = random.sample(batches, 1)[0]
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        a = trainA[start:end]
                        q_a = trainqA[start:end]
                        cost_t_primary = self.model.batch_fit(s, q, a) # anet with primary update

                        total_cost +=  cost_t_aux + cost_t_primary
                        if count % 100 == 0:
                            print("count", count, "aux", cost_t_aux, "primary", cost_t_primary)
                            # print("Aux_gate", aux_gate)
                    print("Ratio of gate_r1/r1: ", gate_r1/n_r1_train, "Ratio of gate_joint_r/joint_r", gate_joint_r/n_r_train)
                elif self.gated_qnet:
                    count = 0
                    for r_start, r_end in r_batches_r:
                        count += 1
                        if FLAGS.separate_eval:
                            start, end = random.sample(r_batches_p, 1)[0]
                        else:
                            start, end = random.sample(batches, 1)[0]
                        r_s_p = trainS[start:end]
                        r_q_p = trainQ[start:end]
                        r_a_p = trainA[start:end]
                        r_q_a_p = trainqA[start:end]

                        r_s = r_trainS[r_start:r_end]
                        r_q = r_trainQ[r_start:r_end]
                        r_a = r_trainA[r_start:r_end]
                        r_q_a = r_trainqA[r_start:r_end]

                        cost_t_outer, _ = self.model.gated_q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p) #gated qnet update

                        cost_t_aux, aux_gate = self.model.gated_batch_fit(r_s, r_q, r_a) #anet with aux update with related data

                        if FLAGS.separate_eval:
                            start, end = random.sample(p_batches, 1)[0]
                        else:
                            start, end = random.sample(batches, 1)[0]
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        a = trainA[start:end]
                        q_a = trainqA[start:end]
                        cost_t_primary = self.model.batch_fit(s, q, a) # anet with primary update

                        total_cost += cost_t_outer + cost_t_aux + cost_t_primary
                        if count % 100 == 0:
                            print("count", count, "outer", cost_t_outer, "aux", cost_t_aux, "primary", cost_t_primary)
                            print("Aux_gate", aux_gate)
                else:
                    if self.alternate:
                        if t % 2 == 0:
                            for start, end in p_batches:
                                s = trainS[start:end]
                                q = trainQ[start:end]
                                a = trainA[start:end]
                                # q_a = trainqA[start:end]
                                cost_t = self.model.q_batch_fit(s, q, a, None, None, None, None, True)  # primary
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
                                outer_cost_t, aux_cost_t = self.model.q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p, False)  # related
                                cost_t = outer_cost_t
                                # print('outer_cost', outer_cost_t, 'aux_cost', aux_cost_t)
                                total_cost += cost_t
                    else:
                        count = 0
                        for start, end in p_batches:
                            count +=1
                            s = trainS[start:end]
                            q = trainQ[start:end]
                            a = trainA[start:end]
                            q_a = trainqA[start:end]
                            cost_t = self.model.q_batch_fit(s, q, a, None, None, None, None, True)  # primary
                            if count%100 == 0:
                                print('primary cost', cost_t)
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
                            outer_cost_t, aux_cost_t = self.model.q_batch_fit(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p, r_a_p, False)  # related
                            cost_t = outer_cost_t
                            if count%100 == 0:
                                print('outer_cost', outer_cost_t, 'aux_cost', aux_cost_t)
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
                if self.has_qnet and self.transform_qnet:
                    train_loss = self.batch_predict_qt(trainS, trainQ, n_train)
                    train_acc = train_loss
                elif self.has_qnet and self.transform_anet:
                    train_loss = self.batch_predict_at(r_trainS, r_trainQ, n_r_train)
                    train_acc = train_loss
                elif self.has_qnet and self.only_related:
                    train_preds = self.batch_predict(r_trainS,r_trainQ,n_r_train, 'qnet')
                    train_acc = metrics.accuracy_score(np.array(train_preds), r_trainA)
                elif self.has_qnet and not self.only_aux and not self.only_primary:
                    train_preds = self.batch_predict(trainS[:int(n_train/2)],trainQ[:int(n_train/2)],int(n_train/2), 'anet')
                    train_acc = metrics.accuracy_score(np.array(train_preds), trainA[:int(n_train/2)])
                else:
                    train_preds = self.batch_predict(trainS,trainQ,n_train, 'anet')
                    train_acc = metrics.accuracy_score(np.array(train_preds), trainA)

                if self.has_qnet and self.transform_qnet:
                    val_loss = self.batch_predict_qt(valS, valQ, n_val)
                    val_acc = val_loss
                elif self.has_qnet and self.transform_anet:
                    val_loss = self.batch_predict_at(r_valS, r_valQ, n_r_val)
                    val_acc = val_loss
                elif self.has_qnet and self.only_related:
                    val_preds = self.batch_predict(r_valS, r_valQ, n_r_val, 'qnet')
                    val_acc = metrics.accuracy_score(val_preds, r_valA)
                elif self.has_qnet and FLAGS.only_gated_qnet:
                    val_preds, val_ans = self.batch_predict_gated_outer(r_trainS, r_trainQ, r_trainA, r_trainqA, n_r_train, valS, valQ, valA, n_val)
                    val_acc = metrics.accuracy_score(val_preds, val_ans)
                else:
                    val_preds = self.batch_predict(valS,valQ,n_val, 'anet')
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

                if self.transform_qnet or self.transform_anet:
                    if val_loss < best_validation_loss:
                        best_validation_loss = val_loss
                        self.saver.save(self.sess, self.model_dir + 'model.ckpt',
                                        global_step=t)
                        print("new model stored")
                else:
                    if val_acc > best_validation_accuracy:
                        best_validation_accuracy=val_acc
                        best_validation_epoch = t
                        self.saver.save(self.sess,self.model_dir+'model.ckpt',
                                        global_step=t)
                        print("new model stored")
                print('Best Validation Accuracy:', best_validation_accuracy)
                print('Best Validation Epoch:', best_validation_epoch)

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
        
        test_preds = self.batch_predict(testS, testQ, n_test, 'anet')
        test_acc = metrics.accuracy_score(test_preds, testA)
        print("Anet Testing Accuracy on primary task:", test_acc)

        if self.has_qnet and self.only_related:
            r_testS, r_testQ, r_testA, _ = vectorize_data(
                self.r_testData, self.word_idx, self.sentence_size, self.r_candidate_sentence_size,
                self.batch_size, self.r_n_cand, self.memory_size)
            n_r_test = len(r_testS)

            test_preds = self.batch_predict(r_testS, r_testQ, n_r_test, 'qnet')
            test_acc = metrics.accuracy_score(test_preds, r_testA)
            print("Qnet Testing Accuracy on related tasks:", test_acc)

        # # Un-comment below to view correct responses and predictions 
        # print(testA)
        # for pred in test_preds:
        #    print(pred, self.indx2candid[pred])

    def batch_predict(self,S,Q,n, type=None):
        """Predict answers over the passed data in batches.

        Args:
            S: Tensor (None, memory_size, sentence_size)
            Q: Tensor (None, sentence_size)
            n: int

        Returns:
            preds: Tensor (None, vocab_size)
        """
        if type == 'qnet':
            predict_qnet = True
        else:
            predict_qnet = False
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            pred = self.model.predict(s, q, predict_qnet)
            preds += list(pred)
        return preds

    def batch_predict_gated_outer(self,r_S,r_Q,r_A, r_qA, r_n, S, Q, A, n):
        """Predict answers over the passed data in batches.

        Args:
            S: Tensor (None, memory_size, sentence_size)
            Q: Tensor (None, sentence_size)
            n: int

        Returns:
            preds: Tensor (None, vocab_size)
        """
        if FLAGS.inner_steps == 1:
            preds = []
            ans = []
            for start in range(0, r_n, self.batch_size):
                end = start + self.batch_size
                for start_p in range(0, n, self.batch_size):
                    end_p = start_p + self.batch_size
                    r_s_p = S[start_p:end_p]
                    r_q_p = Q[start_p:end_p]
                    r_a_p = A[start_p:end_p]
                    # r_q_a_p = trainqA[start_p:end_p]

                    r_s = r_S[start:end]
                    r_q = r_Q[start:end]
                    r_a = r_A[start:end]
                    r_q_a = r_qA[start:end]

                    pred = self.model.predict_gated_outer(r_s, r_q, r_a, r_q_a, r_s_p, r_q_p)  # gated qnet update
                    preds += list(pred)
                    ans += list(r_a_p)
        else:
            preds = []
            ans = []

            r_s_list = []
            r_q_list = []
            r_a_list = []
            r_q_a_list = []
            count  = 0
            for r_start in range(0, r_n, self.batch_size):
                count += 1
                r_end = r_start + self.batch_size

                r_s = r_S[r_start:r_end]
                r_q = r_Q[r_start:r_end]
                r_a = r_A[r_start:r_end]
                r_q_a = r_qA[r_start:r_end]

                r_s_list.append(r_s)
                r_q_list.append(r_q)
                r_a_list.append(r_a)
                r_q_a_list.append(r_q_a)

                if count % FLAGS.inner_steps == 0:
                    for start in range(0,n,self.batch_size):
                        end = start + self.batch_size
                        r_s_p = S[start:end]
                        r_q_p = Q[start:end]
                        r_a_p = A[start:end]

                        # used if outer_r_weight > 0
                        r_start1 = random.sample(range(0, r_n, self.batch_size),1)[0]
                        r_end1 = r_start1 + self.batch_size
                        r_s1 = r_S[r_start1:r_end1]
                        r_q1 = r_Q[r_start1:r_end1]
                        r_a1 = r_A[r_start1:r_end1]
                        r_q_a1 = r_qA[r_start1:r_end1]

                        pred = self.model.predict_gated_outer_list(np.asarray(r_s_list), np.asarray(r_q_list), np.asarray(r_a_list),
                                                                           np.asarray(r_q_a_list), r_s1, r_q1, r_a1, r_q_a1, r_s_p, r_q_p, r_a_p)  # gated qnet update
                        preds += list(pred)
                        ans += list(r_a_p)


                    r_s_list = []
                    r_q_list = []
                    r_a_list = []
                    r_q_a_list = []

        return preds, ans

    def batch_predict_qt(self,S,Q,n):
        """Predict answers over the passed data in batches.

        Args:
            S: Tensor (None, memory_size, sentence_size)
            Q: Tensor (None, sentence_size)
            n: int

        Returns:
            preds: Tensor (None, vocab_size)
        """
        total_loss = 0
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            loss = self.model.predict_qt(s, q)
            total_loss += loss
        return total_loss

    def batch_predict_at(self,S,Q,n):
        """Predict answers over the passed data in batches.

        Args:
            S: Tensor (None, memory_size, sentence_size)
            Q: Tensor (None, sentence_size)
            n: int

        Returns:
            preds: Tensor (None, vocab_size)
        """
        total_loss = 0
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            loss = self.model.predict_at(s, q)
            total_loss += loss
        return total_loss

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':

    model_dir = 'model/' + str(FLAGS.task_id) + '/' + FLAGS.model_dir
    result_dir = 'result/' + str(FLAGS.task_id) + '/' + FLAGS.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    chatbot = chatBot(FLAGS.data_dir, FLAGS.r_data_dir, model_dir, result_dir, FLAGS.task_id, FLAGS.r_task_id, OOV=FLAGS.OOV,
                      has_qnet=FLAGS.has_qnet, batch_size=FLAGS.batch_size, memory_size=FLAGS.memory_size, random_state=FLAGS.random_state,
                      epochs=FLAGS.epochs, hops=FLAGS.hops, save_vocab=FLAGS.save_vocab,
                      load_vocab=FLAGS.load_vocab, learning_rate=FLAGS.learning_rate,
                      embedding_size=FLAGS.embedding_size, evaluation_interval=FLAGS.evaluation_interval,
                      alternate=FLAGS.alternate, only_aux=FLAGS.only_aux, aux_opt=FLAGS.aux_opt,
                      aux_learning_rate=FLAGS.aux_learning_rate, outer_learning_rate=FLAGS.outer_learning_rate,
                      epsilon=FLAGS.epsilon, only_primary=FLAGS.only_primary, max_grad_norm=FLAGS.max_grad_norm,
                      aux_nonlin=FLAGS.aux_nonlin, m_series=FLAGS.m_series, only_related=FLAGS.only_related,
                      transform_qnet=FLAGS.transform_qnet, transform_anet=FLAGS.transform_anet,
                      primary_and_related=FLAGS.primary_and_related, gated_qnet=FLAGS.gated_qnet, outer_r_weight=FLAGS.outer_r_weight)

    if FLAGS.train:
        chatbot.train()
    else:
        chatbot.test()
    
    chatbot.close_session()
