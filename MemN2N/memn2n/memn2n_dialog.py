from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime


class MemN2NDialog(object):
    """End-To-End Memory Network."""
    def __init__(self, has_qnet, batch_size, vocab_size, candidates_size,
                 sentence_size, embedding_size, candidates_vec, candidate_sentence_size,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 outer_optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 aux_optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='anet',
                 q_name = 'qnet',
                 task_id=1,
                 inner_lr=0.01):
        """Creates an End-To-End Memory Network

        Args:
            has_qnet: has question network

            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). 
            The nil word one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences 
            should be padded to this length. If padding is required it should be 
            done with nil one-hot encoding (0).

            candidates_size: The size of candidates

            memory_size: The max size of the memory. Since Tensorflow currently 
            does not support jagged arrays all memories must be padded to this 
            length. If padding is required, the extra memories should be empty 
            memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            candidates_vec: The numpy array of candidates encoding.

            candidate_sentence_size: candidate_sentence_size.

            hops: The number of hops. A hop consists of reading and addressing 
            a memory slot. Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). 
            Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._has_qnet = has_qnet
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._candidate_sentence_size = candidate_sentence_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._outer_opt = outer_optimizer
        self._aux_opt = aux_optimizer
        self._name = name
        self._q_name = q_name
        self._candidates=candidates_vec
        self.inner_lr = inner_lr

        self._build_inputs()
        weights_anet, weights_anet_pred, weights_qnet = self._build_vars()
        weights = {**weights_anet, **weights_anet_pred, **weights_qnet}
        
        # Define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(task_id),'summary_output', timestamp)
        
        # Calculate cross entropy
        # dimensions: (batch_size, candidates_size)
        if self._has_qnet:
            logits, u_k = self._inference(weights, self._stories, self._queries)
            ques_targ, ans_targ = self._q_inference(weights, self._stories, self._queries, self._q_answers)
            # aux_mse = tf.losses.mean_squared_error(labels=ans_targ, predictions=u_k)
            aux_mse = tf.reduce_mean(tf.reduce_sum(tf.square(ans_targ - u_k),1))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self._answers, name="cross_entropy")
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        else:
            logits, _ = self._inference(weights, self._stories, self._queries)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self._answers, name="cross_entropy")
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # Define loss op
        if self._has_qnet:
            inner_loss_op = aux_mse
            loss_op = cross_entropy_sum
        else:
            loss_op = cross_entropy_sum


        if self._has_qnet:
            inner_grads = tf.gradients(inner_loss_op, list(weights_anet.values()))
            inner_grads = [tf.clip_by_norm(grad, self._max_grad_norm) for grad in inner_grads]
            inner_nil_grads = []
            for g, v in zip(inner_grads, list(weights_anet.values())):
                if v.name in self._nil_vars:
                    inner_nil_grads.append(zero_nil_slot(g))
                else:
                    inner_nil_grads.append(g)
            inner_gradients = dict(zip(weights_anet.keys(), inner_nil_grads))
            fast_anet_weights = dict(
                zip(weights_anet.keys(), [
                    weights_anet[key] - self.inner_lr * inner_gradients[key]
                    for key in weights_anet.keys()
                ]))

            outer_logits, _ = self._inference({**fast_anet_weights, **weights_anet_pred}, self._p_stories, self._p_queries)
            outer_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outer_logits, labels=self._p_answers, name="outer_cross_entropy")
            outer_cross_entropy_sum = tf.reduce_sum(outer_cross_entropy, name="outer_cross_entropy_sum")
            outer_loss_op = outer_cross_entropy_sum

            # update of qnet with related task data
            qnet_params = tf.trainable_variables(self._q_name)
            outer_grads_and_vars = self._outer_opt.compute_gradients(outer_loss_op, qnet_params)
            outer_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                                for g,v in outer_grads_and_vars]
            # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
            outer_nil_grads_and_vars = []
            for g, v in outer_grads_and_vars:
                if v.name in self._q_nil_vars:
                    outer_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    outer_nil_grads_and_vars.append((g, v))
            outer_train_op = self._outer_opt.apply_gradients(outer_nil_grads_and_vars, name="outer_train_op")

            # update with auxiliary task
            aux_grads_and_vars = self._aux_opt.compute_gradients(inner_loss_op, list(weights_anet.values()))
            aux_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                                    for g, v in aux_grads_and_vars]
            # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
            aux_nil_grads_and_vars = []
            for g, v in aux_grads_and_vars:
                if v.name in self._nil_vars:
                    aux_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    aux_nil_grads_and_vars.append((g, v))
            aux_train_op = self._aux_opt.apply_gradients(aux_nil_grads_and_vars, name="aux_train_op")

            # update with primary task data
            anet_params = tf.trainable_variables(self._name)
            grads_and_vars = self._opt.compute_gradients(loss_op, anet_params)
            # [print(g,v) for g,v in grads_and_vars]
            grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                                for g,v in grads_and_vars]
            # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
            nil_grads_and_vars = []
            for g, v in grads_and_vars:
                if v.name in self._nil_vars:
                    nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    nil_grads_and_vars.append((g, v))
            train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        else:
            # Gradient pipeline
            anet_params = tf.trainable_variables(self._name)
            grads_and_vars = self._opt.compute_gradients(loss_op, anet_params)
            [print(g,v) for g,v in grads_and_vars]
            grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                                for g,v in grads_and_vars]
            # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
            nil_grads_and_vars = []
            for g, v in grads_and_vars:
                if v.name in self._nil_vars:
                    nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    nil_grads_and_vars.append((g, v))
            train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # Define predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # Assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        self.graph_output = self.loss_op

        if self._has_qnet:
            self.inner_loss_op = inner_loss_op
            self.outer_loss_op = outer_loss_op
            self.outer_train_op = outer_train_op
            self.aux_train_op = aux_train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        """Define input placeholders"""
        self._stories = tf.placeholder(
            tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(
            tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")
        if self._has_qnet:
            self._q_answers = tf.placeholder(
                tf.int32, [None, self._sentence_size], name="q_answers")
            self._p_stories = tf.placeholder(
                tf.int32, [None, None, self._sentence_size], name="p_stories")
            self._p_queries = tf.placeholder(
                tf.int32, [None, self._sentence_size], name="p_queries")
            self._p_answers = tf.placeholder(tf.int32, [None], name="p_answers")

    def _build_vars(self):
        """Define trainable variables"""
        weights_anet = {}
        weights_anet_pred = {}
        weights_qnet = {}
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            # self.A = tf.Variable(A, name="A")
            weights_anet['A'] = tf.Variable(A, name="A")
            # self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            weights_anet['H'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            W = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            # self.W = tf.Variable(W, name="W")
            weights_anet_pred['W'] = tf.Variable(W, name="W")
        self._nil_vars = set([weights_anet['A'].name, weights_anet_pred['W'].name])
        self._q_nil_vars = []
        if self._has_qnet:
            with tf.variable_scope(self._q_name):
                nil_word_slot = tf.zeros([1, self._embedding_size])
                q_A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                # self.q_A = tf.Variable(q_A, name="q_A")
                weights_qnet['q_A'] = tf.Variable(q_A, name="q_A")
                # self.q_H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="q_H")
                weights_qnet['q_H'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="q_H")
                q_W = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                # self.q_W = tf.Variable(q_W, name="q_W") # encoding the answers
                weights_qnet['q_W'] = tf.Variable(q_W, name="q_W") # encoding the answers
                # self.ques_W = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ques_W")
                # weights_qnet['ques_W'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ques_W")
                # self.ans_W = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ans_W")
                weights_qnet['ans_W'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ans_W")
            self._q_nil_vars = set([weights_qnet['q_A'].name, weights_qnet['q_W'].name])
        return weights_anet, weights_anet_pred, weights_qnet


    def _inference(self, weights, stories, queries):
        """Forward pass through the model: Answer network"""
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(weights['A'], queries)  # Queries vector
            
            # Initial state of memory controller for conversation history
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]

            # Iterate over memory for number of hops
            for count in range(self._hops):
                m_emb = tf.nn.embedding_lookup(weights['A'], stories)  # Stories vector
                m = tf.reduce_sum(m_emb, 2)  # Conversation history memory 
                
                # Hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                
                # # Uncomment below to view attention values over memories during inference:
                # probs = tf.Print(
                #     probs, ['memory', count, tf.shape(probs), probs], summarize=200)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                
                # Compute returned vector
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Update controller state
                u_k = tf.matmul(u[-1], weights['H']) + o_k
                
                # Apply nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            candidates_emb=tf.nn.embedding_lookup(weights['W'], self._candidates)
            candidates_emb_sum=tf.reduce_sum(candidates_emb,1)

        return tf.matmul(u_k, tf.transpose(candidates_emb_sum)), u_k

    def _q_inference(self, weights, stories, queries, q_answers):
        if self._has_qnet:
            #Forward pass through the model Question network
            with tf.variable_scope(self._q_name):
                q_q_emb = tf.nn.embedding_lookup(weights['q_A'], queries)  # Queries vector
                q_a_emb = tf.nn.embedding_lookup(weights['q_W'], q_answers) # answers vector

                # Initial state of memory controller for conversation history
                q_q_u_0 = tf.reduce_sum(q_q_emb, 1)
                q_a_u_0 = tf.reduce_sum(q_a_emb, 1)
                q_u = [q_q_u_0 + q_a_u_0]

                # Iterate over memory for number of hops
                for count in range(self._hops):
                    q_m_emb = tf.nn.embedding_lookup(weights['q_A'], stories)  # Stories vector
                    q_m = tf.reduce_sum(q_m_emb, 2)  # Conversation history memory

                    # Hack to get around no reduce_dot
                    q_u_temp = tf.transpose(tf.expand_dims(q_u[-1], -1), [0, 2, 1])
                    q_dotted = tf.reduce_sum(q_m * q_u_temp, 2)

                    # Calculate probabilities
                    q_probs = tf.nn.softmax(q_dotted)

                    # # Uncomment below to view attention values over memories during inference:
                    # probs = tf.Print(
                    #     probs, ['memory', count, tf.shape(probs), probs], summarize=200)

                    q_probs_temp = tf.transpose(tf.expand_dims(q_probs, -1), [0, 2, 1])
                    q_c_temp = tf.transpose(q_m, [0, 2, 1])

                    # Compute returned vector
                    q_o_k = tf.reduce_sum(q_c_temp * q_probs_temp, 2)

                    # Update controller state
                    q_u_k = tf.matmul(q_u[-1], weights['q_H']) + q_o_k

                    # Apply nonlinearity
                    if self._nonlin:
                        q_u_k = self._nonlin(q_u_k)

                    q_u.append(q_u_k)

                # q_candidates_emb = tf.nn.embedding_lookup(self.q_W, self._candidates)
                # q_candidates_emb_sum = tf.reduce_sum(q_candidates_emb, 1)

                # q_question_targ = tf.matmul(q_u_k, weights['ques_W'])
                q_answer_targ = tf.matmul(q_u_k, weights['ans_W'])

        return [], q_answer_targ


    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self._stories: stories, self._queries: queries,
                     self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def q_batch_fit(self, stories, queries, answers, q_answers, p_stories=None, p_queries=None, p_answers=None, primary=True):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
            q_answers: Tensor (None, sentence_size)
            p_*: Primary tasks stories, queries and answers

        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        if primary:
            feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
            loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
            return loss
        else:
            feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._q_answers: q_answers,
                         self._p_stories: p_stories, self._p_queries: p_queries, self._p_answers: p_answers}
            outer_loss, _, inner_loss, _ = self._sess.run([self.outer_loss_op, self.outer_train_op,
                                     self.inner_loss_op, self.aux_train_op], feed_dict=feed_dict)
            return outer_loss, inner_loss

    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)


def zero_nil_slot(t, name=None):
    """Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

