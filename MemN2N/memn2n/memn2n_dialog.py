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
                 max_grad_norm=0.5,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
                 outer_optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
                 aux_optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
                 session=tf.Session(),
                 name='anet',
                 q_name = 'qnet',
                 task_id=1,
                 inner_lr=0.01,
                 aux_opt_name='adam',
                 alpha=0.9,
                 epsilon=1e-8,
                 aux_nonlin=None,
                 m_series=False,
                 r_candidates_vec=None):
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

            aux_opt_name: optimizer for auxiliary task update

            alpha: decay or rmsprop

            epsilon: epsilon of rmsprop and adam.

            aux_nonlin: non-linearity at the end of aux pred/target

            m_series: if set m_series
        """

        self._has_qnet = has_qnet
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._candidate_sentence_size = candidate_sentence_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._qnet_hops = 3
        self._gated_qnet_hops = 1
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._outer_opt = outer_optimizer
        self._aux_opt = aux_optimizer
        self._name = name
        self._q_name = q_name
        self._gated_q_name = 'gated_qnet'
        self._candidates=candidates_vec
        self._inner_lr = inner_lr
        self._aux_opt_name = aux_opt_name
        self._alpha = alpha
        self._epsilon = epsilon
        self._aux_nonlin = aux_nonlin
        self._m_series = m_series
        self._r_candidates=r_candidates_vec
        self._q_opt = tf.train.AdamOptimizer(learning_rate=1e-3, name='q_opt')
        self._qt_opt = tf.train.AdamOptimizer(learning_rate=1e-3, name='qt_opt')
        self._at_opt = tf.train.AdamOptimizer(learning_rate=1e-3, name='at_opt')
        self._r_opt = tf.train.AdamOptimizer(learning_rate=1e-3, name='r_opt')
        self._r_gated_opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3, name='r_gated_opt') #TODO
        self._gated_outer_opt = tf.train.AdamOptimizer(learning_rate=1e-3, name='gated_outer_opt')

        # if self._has_qnet:
        #     self._shared_context_w = True
        #     self._shared_answer_w = True
        # else:
        self._shared_context_w = False
        self._shared_answer_w = False

        self._build_inputs()
        weights_anet, weights_anet_pred, weights_anet_aux, weights_qnet, weights_anet_qnet, weights_anet_pred_qnet, weights_qnet_aux, weights_gated_qnet = self._build_vars()
        weights = {**weights_anet, **weights_anet_pred, **weights_anet_aux, **weights_qnet, **weights_anet_qnet, **weights_anet_pred_qnet, **weights_qnet_aux, **weights_gated_qnet}
        
        # Define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(task_id),'summary_output', timestamp)
        
        # Calculate cross entropy
        # dimensions: (batch_size, candidates_size)
        if self._has_qnet:
            logits, u_k_aux, r_logits = self._inference(weights, self._stories, self._queries) #for m_series u_k_aux is same as u_k
            if self._m_series:
                q_logits, ans_targ = self._q_inference(weights, self._stories, self._queries)
            else:
                q_logits, ans_targ = self._q_inference(weights, self._stories, self._queries, self._q_answers)

            aux_gate = self._gated_q_inference(weights, self._stories, self._queries)

            # aux_mse = tf.losses.mean_squared_error(labels=ans_targ, predictions=u_k)
            aux_mse = tf.reduce_mean(tf.reduce_sum(tf.square(ans_targ - u_k_aux),1))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self._answers, name="cross_entropy")
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

            r_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=r_logits, labels=self._answers, name="r_cross_entropy")
            r_cross_entropy_sum = tf.reduce_sum(r_cross_entropy, name="r_cross_entropy_sum")

            r_gated_cross_entropy = aux_gate * r_cross_entropy
            r_gated_cross_entropy_sum = tf.reduce_sum(r_gated_cross_entropy, name="r_gated_cross_entropy_sum")

            q_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=q_logits, labels=self._answers, name="q_cross_entropy")
            q_cross_entropy_sum = tf.reduce_sum(q_cross_entropy, name="q_cross_entropy_sum")

            qt_mse_loss = tf.losses.mean_squared_error(labels=u_k_aux, predictions=ans_targ)
            at_mse_loss = tf.losses.mean_squared_error(labels=ans_targ, predictions=u_k_aux)

        else:
            logits, _, _ = self._inference(weights, self._stories, self._queries)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self._answers, name="cross_entropy")
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # Define loss op
        if self._has_qnet:
            inner_loss_op = aux_mse
            loss_op = cross_entropy_sum
            r_loss_op = r_cross_entropy_sum
            r_gated_loss_op = r_gated_cross_entropy_sum
            q_loss_op = q_cross_entropy_sum
            qt_loss_op = qt_mse_loss
            at_loss_op = at_mse_loss
        else:
            loss_op = cross_entropy_sum

        if self._has_qnet:
            # update with auxiliary task
            aux_grads = tf.gradients(inner_loss_op, list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values()))
            aux_grads_and_vars = zip(aux_grads, list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values()))
            aux_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                                  for g, v in aux_grads_and_vars]
            aux_nil_grads_and_vars = []
            for g, v in aux_grads_and_vars:
                if v.name in self._nil_vars:
                    aux_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    aux_nil_grads_and_vars.append((g, v))
            aux_train_op = self._aux_opt.apply_gradients(aux_nil_grads_and_vars, name="aux_train_op")

            #simulate the auxiliary update
            inner_grads = tf.gradients(inner_loss_op, list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values()))
            inner_grads = [tf.clip_by_norm(grad, self._max_grad_norm) for grad in inner_grads]
            inner_nil_grads = []
            for g, v in zip(inner_grads, list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values())):
                if v.name in self._nil_vars:
                    inner_nil_grads.append(zero_nil_slot(g))
                else:
                    inner_nil_grads.append(g)
            if self._aux_opt_name == 'rms':
                rmss = [self._aux_opt.get_slot(var, 'rms') for var in (list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values()))]
                fast_anet_weights = {}
                for grad, rms, var, var_name in zip(inner_nil_grads, rmss, (list(weights_anet.values())+ list(weights_anet_aux.values()) + list(weights_anet_qnet.values())),
                                                    (list(weights_anet.keys()) + list(weights_anet_aux.keys()) + list(weights_anet_qnet.keys()))):
                    ms = rms + (tf.square(grad) - rms) * (1 - self._alpha)
                    fast_anet_weights[var_name] = var - self._inner_lr * grad / tf.sqrt(ms + self._epsilon)
            elif self._aux_opt_name == 'sgd':
                fast_anet_weights = {}
                for grad, var, var_name in zip(inner_nil_grads, (list(weights_anet.values())+ list(weights_anet_aux.values()) + list(weights_anet_qnet.values())),
                                               (list(weights_anet.keys()) + list(weights_anet_aux.keys()) + list(weights_anet_qnet.keys()))):
                    fast_anet_weights[var_name] = var - self._inner_lr * grad
            elif self._aux_opt_name == 'adam':
                # ms = [self._aux_opt.get_slot(var, 'm') for var in (list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values()))]
                # vs = [self._aux_opt.get_slot(var, 'v') for var in (list(weights_anet.values()) + list(weights_anet_aux.values()) + list(weights_anet_qnet.values()))]
                # fast_anet_weights = {}
                # for grad, m, v, var, var_name in zip(inner_nil_grads, ms, vs, (list(weights_anet.values())+ list(weights_anet_aux.values()) + list(weights_anet_qnet.values())),
                #                                (list(weights_anet.keys()) + list(weights_anet_aux.keys()) + list(weights_anet_qnet.keys()))):
                #     fast_anet_weights[var_name] = var - self._aux_opt._lr_t * m / tf.sqrt(v + self._epsilon)
                fast_anet_weights = {}
                for grad, var, var_name in zip(inner_nil_grads, (list(weights_anet.values())+ list(weights_anet_aux.values()) + list(weights_anet_qnet.values())),
                                               (list(weights_anet.keys()) + list(weights_anet_aux.keys()) + list(weights_anet_qnet.keys()))):
                    fast_anet_weights[var_name] = var - self._inner_lr * grad

            outer_logits, _, _ = self._inference({**fast_anet_weights, **weights_anet_pred, **weights_anet_pred_qnet}, self._p_stories, self._p_queries)
            outer_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outer_logits, labels=self._p_answers, name="outer_cross_entropy")
            outer_cross_entropy_sum = tf.reduce_sum(outer_cross_entropy, name="outer_cross_entropy_sum")
            outer_loss_op = outer_cross_entropy_sum

            if not self._m_series:
                # update of qnet
                outer_grads = tf.gradients(outer_loss_op, list(weights_qnet.values()) + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
                outer_grads_and_vars = zip(outer_grads, list(weights_qnet.values()) + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
                outer_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in outer_grads_and_vars]
                outer_nil_grads_and_vars = []
                for g, v in outer_grads_and_vars:
                    if v.name in self._q_nil_vars:
                        outer_nil_grads_and_vars.append((zero_nil_slot(g), v))
                    else:
                        outer_nil_grads_and_vars.append((g, v))
                outer_train_op = self._outer_opt.apply_gradients(outer_nil_grads_and_vars, name="outer_train_op")
            else:
                outer_train_op = None

            # update with primary task data
            grads = tf.gradients(loss_op, list(weights_anet.values()) + list(weights_anet_pred.values()) + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            grads_and_vars = zip(grads, list(weights_anet.values()) + list(weights_anet_pred.values()) + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                                for g,v in grads_and_vars]
            nil_grads_and_vars = []
            for g, v in grads_and_vars:
                if v.name in self._nil_vars:
                    nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    nil_grads_and_vars.append((g, v))
            train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

            # update anet with related task data
            r_grads = tf.gradients(r_loss_op, list(weights_anet.values()) + list(weights_anet_pred.values()) + list(
                weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            r_grads_and_vars = zip(r_grads, list(weights_anet.values()) + list(weights_anet_pred.values()) + list(
                weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            r_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                              for g, v in r_grads_and_vars]
            r_nil_grads_and_vars = []
            for g, v in r_grads_and_vars:
                if v.name in self._nil_vars:
                    r_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    r_nil_grads_and_vars.append((g, v))
            r_train_op = self._r_opt.apply_gradients(r_nil_grads_and_vars, name="r_train_op")

            # update anet with gated related task data
            r_gated_grads = tf.gradients(r_gated_loss_op, list(weights_anet.values()) + list(weights_anet_pred.values()) + list(
                weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            r_gated_grads_and_vars = zip(r_gated_grads, list(weights_anet.values()) + list(weights_anet_pred.values()) + list(
                weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            r_gated_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                              for g, v in r_gated_grads_and_vars]
            r_gated_nil_grads_and_vars = []
            for g, v in r_gated_grads_and_vars:
                if v.name in self._nil_vars:
                    r_gated_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    r_gated_nil_grads_and_vars.append((g, v))
            r_gated_train_op = self._r_gated_opt.apply_gradients(r_gated_nil_grads_and_vars, name="r_gated_train_op")

            # simulate the auxiliary update for anet
            gated_inner_grads = tf.gradients(r_gated_loss_op, list(weights_anet.values()) + list(weights_anet_pred.values())
                                             + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values()))
            gated_inner_grads = [tf.clip_by_norm(grad, self._max_grad_norm) for grad in gated_inner_grads]
            gated_inner_nil_grads = []
            for g, v in zip(gated_inner_grads, list(weights_anet.values()) + list(weights_anet_pred.values())
                                               + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values())):
                if v.name in self._nil_vars:
                    gated_inner_nil_grads.append(zero_nil_slot(g))
                else:
                    gated_inner_nil_grads.append(g)

            gated_fast_anet_weights = {}
            for grad, var, var_name in zip(gated_inner_nil_grads, (list(weights_anet.values()) + list(weights_anet_pred.values())
                                                                   + list(weights_anet_qnet.values()) + list(weights_anet_pred_qnet.values())),
                                           (list(weights_anet.keys()) + list(weights_anet_pred.keys()) + list(weights_anet_qnet.keys())
                                            + list(weights_anet_pred_qnet.keys()))):
                gated_fast_anet_weights[var_name] = var - self._inner_lr * grad


            gated_outer_logits, _, _ = self._inference(gated_fast_anet_weights, self._p_stories, self._p_queries)
            gated_outer_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=gated_outer_logits, labels=self._p_answers, name="gated_outer_cross_entropy")
            gated_outer_cross_entropy_sum = tf.reduce_sum(gated_outer_cross_entropy, name="gated_outer_cross_entropy_sum")
            gated_outer_loss_op = gated_outer_cross_entropy_sum

            # Update qnet (gated)
            gated_outer_grads = tf.gradients(gated_outer_loss_op, list(weights_gated_qnet.values()))
            gated_outer_grads_and_vars = zip(gated_outer_grads,list(weights_gated_qnet.values()))
            gated_outer_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in gated_outer_grads_and_vars]
            gated_outer_nil_grads_and_vars = []
            for g, v in gated_outer_grads_and_vars:
                if v.name in self._gated_q_nil_vars:
                    gated_outer_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    gated_outer_nil_grads_and_vars.append((g, v))
            gated_outer_train_op = self._gated_outer_opt.apply_gradients(gated_outer_nil_grads_and_vars, name="gated_outer_train_op")

            # update qnet with related task data
            q_grads = tf.gradients(q_loss_op, list(weights_qnet.values()))
            q_grads_and_vars = zip(q_grads, list(weights_qnet.values()))
            q_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in q_grads_and_vars]
            q_nil_grads_and_vars = []
            for g, v in q_grads_and_vars:
                if v.name in self._q_nil_vars:
                    q_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    q_nil_grads_and_vars.append((g, v))
            q_train_op = self._q_opt.apply_gradients(q_nil_grads_and_vars, name="q_train_op")

            # update qnet_aux
            qt_grads = tf.gradients(qt_loss_op, list(weights_qnet_aux.values()))
            qt_grads_and_vars = zip(qt_grads, list(weights_qnet_aux.values()))
            qt_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in qt_grads_and_vars]
            qt_nil_grads_and_vars = []
            for g, v in qt_grads_and_vars:
                if v.name in self._q_nil_vars:
                    qt_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    qt_nil_grads_and_vars.append((g, v))
            qt_train_op = self._qt_opt.apply_gradients(qt_nil_grads_and_vars, name="qt_train_op")

            # update anet with qnet_aux
            at_grads = tf.gradients(at_loss_op, list(weights_anet.values()))
            at_grads_and_vars = zip(at_grads, list(weights_anet.values()))
            at_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in at_grads_and_vars]
            at_nil_grads_and_vars = []
            for g, v in at_grads_and_vars:
                if v.name in self._nil_vars:
                    at_nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    at_nil_grads_and_vars.append((g, v))
            at_train_op = self._at_opt.apply_gradients(at_nil_grads_and_vars, name="at_train_op")

            if self._m_series:
                assign_a = weights_anet['A'].assign(weights_qnet['q_A'])
                assign_h = weights_anet['H'].assign(weights_qnet['q_H'])
                assign_w = weights_anet_pred['W'].assign(weights_qnet['q_W'])
                assign_qnet2anet_op = [assign_a, assign_h, assign_w]
            else:
                assign_qnet2anet_op = None
        else:
            # Gradient pipeline
            grads_and_vars = self._opt.compute_gradients(loss_op,
                                                         list(weights_anet.values()) + list(weights_anet_pred.values()) + list(weights_anet_qnet.values())
                                                         + list(weights_anet_pred_qnet.values()))
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
            self.aux_train_op = aux_train_op
            self.outer_train_op = outer_train_op
            self.q_loss_op = q_loss_op
            self.q_train_op = q_train_op

            q_predict_op = tf.argmax(q_logits, 1, name="q_predict_op")
            self.q_predict_op = q_predict_op

            self.qt_loss_op = qt_loss_op
            self.qt_train_op = qt_train_op

            self.at_loss_op = at_loss_op
            self.at_train_op = at_train_op

            self.assign_qnet2anet_op = assign_qnet2anet_op

            self.r_loss_op = r_loss_op
            self.r_train_op = r_train_op

            self.r_gated_loss_op = r_gated_loss_op
            self.r_gated_train_op = r_gated_train_op

            self.gated_outer_loss_op = gated_outer_loss_op
            self.gated_outer_train_op = gated_outer_train_op

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
        weights_anet = {} # weights only for anet update, aux and primary
        weights_anet_pred = {} # weights only for anet primary update
        weights_anet_aux = {} # weights only for anet aux update
        weights_qnet = {} # weights only for qnet update
        weights_anet_qnet = {} # shared between anet and qnet in all updates
        weights_anet_pred_qnet = {} # weights for qnet and anet primary update
        weights_qnet_aux = {}
        weights_gated_qnet = {}

        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            if self._shared_context_w:
                weights_anet_qnet['A'] = tf.Variable(A, name="A")
            else:
                weights_anet['A'] = tf.Variable(A, name="A")
            weights_anet['H'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            if self._has_qnet and not self._m_series:
                weights_anet_aux['H_aux'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H_aux")
                weights_anet_pred['H_p'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H_p")
            W = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            if self._shared_answer_w:
                weights_anet_pred_qnet['W'] = tf.Variable(W, name="W")
            else:
                weights_anet_pred['W'] = tf.Variable(W, name="W")
        if self._shared_context_w and self._shared_answer_w: #TODO If only of them is True
            self._nil_vars = set([weights_anet_qnet['A'].name, weights_anet_pred_qnet['W'].name])
        else:
            self._nil_vars = set([weights_anet['A'].name, weights_anet_pred['W'].name])

        self._q_nil_vars = []
        if self._has_qnet:
            with tf.variable_scope(self._q_name):
                nil_word_slot = tf.zeros([1, self._embedding_size])
                if not self._shared_context_w:
                    q_A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                    weights_qnet['q_A'] = tf.Variable(q_A, name="q_A")
                weights_qnet['q_H'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="q_H")
                if not self._shared_answer_w:
                    q_W = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                    weights_qnet['q_W'] = tf.Variable(q_W, name="q_W") # encoding the answers
                # weights_qnet['ques_W'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ques_W")
                if self._m_series:
                    weights_qnet_aux['ans_W'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ans_W")
                else:
                    weights_qnet['ans_W'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="ans_W")

            with tf.variable_scope(self._gated_q_name):
                gated_q_A = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])])
                weights_gated_qnet['gated_q_A'] = tf.Variable(gated_q_A, name="gated_q_A")
                weights_gated_qnet['gated_q_H'] = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="gated_q_H")
                weights_gated_qnet['gated_q_W'] = tf.Variable(self._init([self._embedding_size, 1]), name="gated_q_W")

        if not (self._shared_context_w and self._shared_answer_w):  # TODO If only of them is True
            self._q_nil_vars = set([weights_qnet['q_A'].name, weights_qnet['q_W'].name]) #TODO

        self._gated_q_nil_vars = set([weights_gated_qnet['gated_q_A'].name])  # TODO

        return weights_anet, weights_anet_pred, weights_anet_aux, weights_qnet, weights_anet_qnet, weights_anet_pred_qnet, weights_qnet_aux, weights_gated_qnet


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

            r_candidates_emb=tf.nn.embedding_lookup(weights['W'], self._r_candidates)
            r_candidates_emb_sum=tf.reduce_sum(r_candidates_emb,1)

            if self._has_qnet and not self._m_series:
                u_k_p = tf.matmul(u_k, weights['H_p'])
                if self._aux_nonlin == 'arctan':
                    u_k_aux = tf.math.atan(tf.matmul(u_k, weights['H_aux']))
                else:
                    u_k_aux = tf.math.atan(tf.matmul(u_k, weights['H_aux']))
                return tf.matmul(u_k_p, tf.transpose(candidates_emb_sum)), u_k_aux, tf.matmul(u_k, tf.transpose(r_candidates_emb_sum))
            else:
                return tf.matmul(u_k, tf.transpose(candidates_emb_sum)), u_k, tf.matmul(u_k, tf.transpose(r_candidates_emb_sum))

    def _q_inference(self, weights, stories, queries, q_answers=None):
        if self._has_qnet:
            #Forward pass through the model Question network
            with tf.variable_scope(self._q_name):
                if self._shared_context_w:
                    q_q_emb = tf.nn.embedding_lookup(weights['A'], queries)  # Queries vector
                else:
                    q_q_emb = tf.nn.embedding_lookup(weights['q_A'], queries)  # Queries vector
                if not self._m_series:
                    if self._shared_answer_w:
                        q_a_emb = tf.nn.embedding_lookup(weights['W'], q_answers) # answers vector
                    else:
                        q_a_emb = tf.nn.embedding_lookup(weights['q_W'], q_answers) # answers vector

                # Initial state of memory controller for conversation history
                q_q_u_0 = tf.reduce_sum(q_q_emb, 1)
                if not self._m_series:
                    q_a_u_0 = tf.reduce_sum(q_a_emb, 1)
                    q_u = [q_q_u_0 + q_a_u_0]
                else:
                    q_u = [q_q_u_0]

                # Iterate over memory for number of hops
                for count in range(self._qnet_hops):
                    if self._shared_context_w:
                        q_m_emb = tf.nn.embedding_lookup(weights['A'], stories)  # Stories vector
                    else:
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


                q_candidates_emb = tf.nn.embedding_lookup(weights['q_W'], self._r_candidates)
                q_candidates_emb_sum = tf.reduce_sum(q_candidates_emb, 1)

                # q_question_targ = tf.matmul(q_u_k, weights['ques_W'])
                if self._aux_nonlin == 'arctan':
                    q_answer_targ = tf.math.atan(tf.matmul(q_u_k, weights['ans_W']))
                else:
                    q_answer_targ = tf.matmul(q_u_k, weights['ans_W'])

        return tf.matmul(q_u_k, tf.transpose(q_candidates_emb_sum)), q_answer_targ

    def _gated_q_inference(self, weights, stories, queries, q_answers=None):
        if self._has_qnet:
            #Forward pass through the model Question network
            with tf.variable_scope(self._gated_q_name):
                q_q_emb = tf.nn.embedding_lookup(weights['gated_q_A'], queries)  # Queries vectorr

                # Initial state of memory controller for conversation history
                q_q_0 = tf.reduce_sum(q_q_emb, 1)
                q_u = [q_q_0]

                # Iterate over memory for number of hops
                for count in range(self._gated_qnet_hops):

                    q_m_emb = tf.nn.embedding_lookup(weights['gated_q_A'], stories)  # Stories vector

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
                    q_u_k = tf.matmul(q_u[-1], weights['gated_q_H']) + q_o_k

                    # Apply nonlinearity
                    if self._nonlin:
                        q_u_k = self._nonlin(q_u_k)

                    q_u.append(q_u_k)

                gate_weight = tf.nn.sigmoid(tf.matmul(q_u_k, weights['gated_q_W']))

        return gate_weight


    def batch_fit(self, stories, queries, answers, primary=True):
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

        if primary:
            loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        else:
            loss, _ = self._sess.run([self.r_loss_op, self.r_train_op], feed_dict=feed_dict)
        return loss

    def gated_batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}

        loss, _ = self._sess.run([self.r_gated_loss_op, self.r_gated_train_op], feed_dict=feed_dict)
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
            inner_loss, _,  outer_loss, _ = self._sess.run([self.inner_loss_op, self.aux_train_op,
                                                             self.outer_loss_op, self.outer_train_op], feed_dict=feed_dict)
            return outer_loss, inner_loss

    def gated_q_batch_fit(self, stories, queries, answers, q_answers, p_stories=None, p_queries=None, p_answers=None):
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

        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._q_answers: q_answers,
                     self._p_stories: p_stories, self._p_queries: p_queries, self._p_answers: p_answers}
        outer_loss, _ = self._sess.run([self.gated_outer_loss_op, self.gated_outer_train_op], feed_dict=feed_dict)

        return outer_loss

    def q_batch_fit_r(self, stories, queries, answers):
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
        loss, _ = self._sess.run([self.q_loss_op, self.q_train_op], feed_dict=feed_dict)
        return loss

    def batch_fit_qt(self, stories, queries, answers):
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
        loss, _ = self._sess.run([self.qt_loss_op, self.qt_train_op], feed_dict=feed_dict)
        return loss

    def batch_fit_at(self, stories, queries, answers):
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
        loss, _ = self._sess.run([self.at_loss_op, self.at_train_op], feed_dict=feed_dict)
        return loss

    def predict_qt(self, stories, queries):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self._stories: stories, self._queries: queries}
        loss = self._sess.run(self.qt_loss_op, feed_dict=feed_dict)
        return loss

    def predict_at(self, stories, queries):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self._stories: stories, self._queries: queries}
        loss = self._sess.run(self.at_loss_op, feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries, predict_qnet):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        if predict_qnet:
            return self._sess.run(self.q_predict_op, feed_dict=feed_dict)
        else:
            return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def copy_qnet2anet(self):
        self._sess.run(self.assign_qnet2anet_op)


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

