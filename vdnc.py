import tensorflow as tf
import numpy as np
import os
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memory import Memory
import utility



def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    # k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,-1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def f_gaussian(muy, lsigma):
    e = tf.random_normal(muy.shape)
    rsig=tf.exp(lsigma / 2)
    z = rsig*e


    return z+muy


class VariationalDNC:
    def __init__(self, controller_class, input_encoder_size, input_decoder_size, output_size,
                 memory_words_num = 256, memory_word_size = 64, memory_read_heads = 4,
                 batch_size = 1, hidden_controller_dim=256, use_emb_encoder=True, use_emb_decoder=True,
                 use_mem=True, decoder_mode=False, emb_size=64,
                 write_protect=False, dual_emb=True,
                 use_teacher=False, attend_dim=0,
                 use_encoder_output=False,
                 pass_encoder_state=True,
                 memory_read_heads_decode=None,
                 enable_drop_out=False, nlayer=1,
                 KL_anneal=True, single_KL=False,
                 name='VDNC', gt_type='bow'):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html
        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        memory_words_num: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        saved_args = locals()
        print("saved_args is", saved_args)
        self.name=name
        self.input_encoder_size = input_encoder_size
        self.input_decoder_size = input_decoder_size
        self.output_size = output_size
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        if memory_read_heads_decode is None:
            self.read_heads_decode = memory_read_heads
        else:
            self.read_heads_decode = memory_read_heads_decode
        self.batch_size = batch_size
        self.unpacked_input_encoder_data = None
        self.unpacked_input_decoder_data = None
        self.unpacked_target_data = None
        self.packed_output = None
        self.packed_kl_losses = None
        self.packed_memory_view_encoder = None
        self.packed_memory_view_decoder = None
        self.decoder_mode = decoder_mode
        self.emb_size = emb_size
        self.emb_size2 = emb_size
        self.dual_emb = dual_emb
        self.use_mem = use_mem
        self.use_emb_encoder = use_emb_encoder
        self.use_emb_decoder = use_emb_decoder
        self.hidden_controller_dim = hidden_controller_dim
        self.attend_dim = attend_dim
        self.use_teacher = use_teacher
        self.teacher_force = tf.placeholder(tf.bool,[None], name='teacher')
        self.use_encoder_output=use_encoder_output
        self.pass_encoder_state=pass_encoder_state
        self.clear_mem = tf.placeholder(tf.bool,None, name='clear_mem')
        self.drop_out_keep = tf.placeholder(tf.float32, shape=(), name='drop_out_keep')
        self.nlayer = nlayer
        self.epochs = tf.placeholder(tf.float32, shape=(), name='epochs')
        self.KL_anneal = KL_anneal
        self.single_KL = single_KL
        self.gt_type=gt_type
        drop_out_v = 1
        if enable_drop_out:
            drop_out_v = self.drop_out_keep


        self.controller_out = self.output_size



        if self.use_emb_encoder is False:
            self.emb_size=input_encoder_size

        if self.use_emb_decoder is False:
            self.emb_size2=input_decoder_size

        if self.attend_dim>0:
            self.W_a = tf.get_variable('W_a', [hidden_controller_dim, self.attend_dim],
                                      initializer=tf.random_normal_initializer(stddev=0.1))

            value_size = self.hidden_controller_dim
            if self.use_mem:
                value_size = self.word_size
            self.U_a = tf.get_variable('U_a', [value_size, self.attend_dim],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
            if self.use_mem:
                self.V_a = tf.get_variable('V_a', [self.read_heads_decode*self.word_size, self.attend_dim],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self.v_a = tf.get_variable('v_a', [self.attend_dim],
                                  initializer=tf.random_normal_initializer(stddev=0.1))

        # DNC (or NTM) should be structurized into 2 main modules:
        # all the graph is setup inside these twos
        self.W_emb_encoder = tf.get_variable('embe_w', [self.input_encoder_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        self.W_emb_decoder = tf.get_variable('embd_w', [self.output_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.emb_size, self.controller_out, self.read_heads,
                                           self.word_size, self.batch_size, use_mem,
                                           hidden_dim=hidden_controller_dim, drop_out_keep=drop_out_v, nlayer=nlayer)
        self.dual_controller = True
        with tf.variable_scope('controller2_scope'):
            if attend_dim == 0 or use_mem:
                self.controller2 = controller_class(self.emb_size2, self.controller_out, self.read_heads_decode,
                                                    self.word_size, self.batch_size, use_mem,
                                                    hidden_dim=hidden_controller_dim, drop_out_keep=drop_out_v,
                                                    nlayer=nlayer, vae_mode=True )
            else:
                self.controller2 = controller_class(self.emb_size2 + hidden_controller_dim, self.controller_out,
                                                    self.read_heads_decode,
                                                    self.word_size, self.batch_size, use_mem,
                                                    hidden_dim=hidden_controller_dim, drop_out_keep=drop_out_v,
                                                    nlayer=nlayer, vae_mode=True)
        self.write_protect = write_protect


        # input data placeholders

        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')

        self.input_encoder = tf.placeholder(tf.float32, [batch_size, None, input_encoder_size], name='input_encoder')

        self.input_decoder = tf.placeholder(tf.float32, [batch_size, None, input_decoder_size], name='input_decoder')

        self.mask = tf.placeholder(tf.bool, [batch_size, None], name='mask')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')# variant length?
        self.decode_length = tf.placeholder(tf.int32, name='decode_length')  # variant length?


        # distribution transformation from memory to posterior space
        size_gt=0
        if self.gt_type=='bow':
            self.size_gt=self.emb_size2
        else:
            self.size_gt=self.hidden_controller_dim*2

        if self.single_KL:
            self.W_recog = tf.get_variable('recog_net', [self.hidden_controller_dim * 2,
                                                         self.hidden_controller_dim * 2],
                                           initializer=tf.random_normal_initializer(stddev=0.1))

            self.W_pior = tf.get_variable('pior_net', [self.hidden_controller_dim * 2,
                                                       self.hidden_controller_dim * 2*self.read_heads_decode],
                                          initializer=tf.random_normal_initializer(stddev=0.1))
            self.W_modew = tf.get_variable('modew_net', [self.hidden_controller_dim * 2,
                                                       self.read_heads_decode],
                                          initializer=tf.random_normal_initializer(stddev=0.1))

        else:
            # distribution transformation from memory to posterior space
            self.WLz_xy = tf.get_variable('WLz_xy', [self.word_size+self.size_gt, self.word_size],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self.bz_xy = tf.get_variable('bz_xy', [self.word_size],
                                        initializer=tf.initializers.random_uniform(-1, 1))
            self.WLz_xy2 = tf.get_variable('WLz_xy2', [self.word_size, self.word_size],
                                          initializer=tf.random_normal_initializer(stddev=0.1))
            self.bz_xy2 = tf.get_variable('bz_xy2', [self.word_size],
                                         initializer=tf.initializers.random_uniform(-1, 1))
            # distribution transformation from memory to prior space
            self.WLz_x = tf.get_variable('WLz_x', [self.word_size, self.word_size],
                                          initializer=tf.random_normal_initializer(stddev=0.1))
            self.bz_x = tf.get_variable('bz_x', [self.word_size],
                                         initializer=tf.initializers.random_uniform(-1, 1))
            self.WLz_x2 = tf.get_variable('WLz_x2', [self.word_size, self.word_size],
                                         initializer=tf.random_normal_initializer(stddev=0.1))
            self.bz_x2 = tf.get_variable('bz_x2', [self.word_size],
                                         initializer=tf.initializers.random_uniform(-1, 1))

            self.Wmixw = tf.get_variable('Wmixw', [self.hidden_controller_dim, self.read_heads_decode],
                                          initializer=tf.random_normal_initializer(stddev=0.1))


            #similar to the case of single Gaussian

            self.WLz_xy_uni = tf.get_variable('WLz_xy_uni', [self.hidden_controller_dim+self.size_gt, self.word_size],
                                          initializer=tf.random_normal_initializer(stddev=0.1))
            self.WLz_xy_uni2 = tf.get_variable('WLz_xy_uni2', [self.word_size, self.word_size],
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            self.WLz_x_uni = tf.get_variable('WLz_x_uni', [self.hidden_controller_dim, self.word_size],
                                         initializer=tf.random_normal_initializer(stddev=0.1))
            self.WLz_x_uni2 = tf.get_variable('WLz_x_uni2', [self.word_size, self.word_size],
                                             initializer=tf.random_normal_initializer(stddev=0.1))

            self.Wz2rnn = tf.Variable(
                tf.random_normal([self.word_size//2, self.word_size], stddev=0.1),
                name='Wz2rnn'
            )


        self.testing_phase = tf.placeholder(tf.bool, name='clear_mem')

        self.build_graph()

    @staticmethod
    def dist2gaussian(dist_vector):
        muy, logsigma = tf.split(dist_vector, num_or_size_splits=2, axis=-1)
        return [muy, logsigma]


    def KL2gauss_log(self, dist1, dist2, N):
        muy1, lsig1 = self.dist2gaussian(dist1)
        muy2, lsig2 = self.dist2gaussian(dist2)
        kl_loss =0.5 * (tf.reduce_sum(
            lsig2-lsig1+tf.exp(lsig1-lsig2)+
            tf.square(muy2-muy1)/tf.exp(lsig2), axis=-1)-N)
        return kl_loss

    def logsumtrick(self, x):
        b = tf.reduce_max(x, axis=-1)
        x = x - tf.tile(tf.expand_dims(b, 1), [1, self.read_heads_decode])
        ls = b + tf.log(tf.reduce_sum(tf.exp(x), axis=-1))
        return ls

    def KLmixgauss_log(self, dist1, dist2s, mixture_w, N):
        muy1, lsig1 = self.dist2gaussian(dist1)
        muy2, lsig2 = self.dist2gaussian(dist2s)
        blsig1=tf.tile(tf.expand_dims(lsig1, 1), [1, self.read_heads_decode, 1])
        bmuy1 = tf.tile(tf.expand_dims(muy1, 1), [1, self.read_heads_decode, 1])
        kl_loss = 0.5 * (tf.reduce_sum(
            lsig2-blsig1+tf.exp(blsig1-lsig2)+
            tf.square(muy2-bmuy1)/tf.exp(lsig2), axis=-1)-N)
        kl_loss=-self.logsumtrick(-kl_loss+tf.log(mixture_w))
        return kl_loss #[batch_size]

    def sample_the_uni(self, dist_vector):
        muy, sigma = self.dist2gaussian(dist_vector)
        return f_gaussian(muy,sigma)

    #just take the mixture mean
    def sample_the_mixture(self, dist_vector, mixture_w):
        muy, sigma = self.dist2gaussian(dist_vector)#[self.batch_size, self.read_heads_decode, self.word_size*2]
        z  = tf.reduce_sum(muy * tf.expand_dims(mixture_w, 2), 1)# [batch_size, word_size]
        return z

    # use gumbel softmax trick to sample the mixture
    def sample_the_mixture_hard(self, dist_vector, mixutre_w):
        muy, sigma = self.dist2gaussian(dist_vector)  # [self.batch_size, self.read_heads_decode, self.word_size*2]
        zs = f_gaussian(muy, sigma)
        temp=1e-4
        oh = gumbel_softmax(mixutre_w, temperature=temp, hard=True) # [self.batch_size, self.read_heads_decode]
        z = tf.reduce_sum(zs * tf.expand_dims(oh, 2), 1)  # [batch_size, word_size]
        return z

    def get_the_posterior_stand_dist(self):
        dist_vector1 = tf.zeros([self.batch_size, self.word_size])
        dist_vector2= tf.ones([self.batch_size, self.word_size])
        dist_vector = tf.concat([dist_vector1, dist_vector2], axis=-1)
        return dist_vector

    def get_the_posterior_dist(self, read_vector_y):
        read_vector_y = tf.reshape(read_vector_y, [self.batch_size, -1])
        dist_vector = tf.matmul(read_vector_y, self.WLz_xy)  #  word size
        dist_vector += self.bz_xy
        dist_vector = tf.nn.relu(dist_vector)
        dist_vector = tf.matmul(dist_vector, self.WLz_xy2)  # word size
        dist_vector += self.bz_xy2
        # dist_vector = tf.nn.relu(dist_vector)
        mean, std = tf.split(dist_vector, num_or_size_splits=2, axis=-1)
        std = tf.nn.softplus(std)
        dist_vector = tf.concat([mean,std], axis=-1)
        return dist_vector

    def get_the_prior_dist(self, read_vector_x, dim):
        dist_vector = tf.reshape(read_vector_x, [-1, dim])
        dist_vector = tf.matmul(dist_vector, self.WLz_x) #  word size
        dist_vector += self.bz_x
        dist_vector = tf.nn.relu(dist_vector)
        dist_vector = tf.matmul(dist_vector, self.WLz_x2)  # word size
        dist_vector += self.bz_x2
        # dist_vector = tf.nn.relu(dist_vector)
        mean, std = tf.split(dist_vector, num_or_size_splits=2, axis=-1)
        std = tf.nn.softplus(std)
        dist_vector = tf.concat([mean, std], axis=-1)
        dist_vector = tf.reshape(dist_vector, [self.batch_size, self.read_heads_decode,-1])
        return dist_vector

    def get_the_posterior_dist_uni(self, read_vector_y):
        read_vector_y = tf.reshape(read_vector_y, [self.batch_size, -1])
        dist_vector = tf.matmul(read_vector_y, self.WLz_xy_uni)  # word size
        dist_vector += self.bz_xy
        dist_vector = tf.nn.relu(dist_vector)
        dist_vector = tf.matmul(dist_vector, self.WLz_xy_uni2)  # word size
        dist_vector += self.bz_xy2
        # dist_vector = tf.nn.relu(dist_vector)
        mean, std = tf.split(dist_vector, num_or_size_splits=2, axis=-1)
        std = tf.nn.softplus(std)
        dist_vector = tf.concat([mean, std], axis=-1)
        return dist_vector

    def get_the_pior_dist_uni(self, dist_vector):
        if not self.single_KL:
            dist_vector = tf.matmul(dist_vector, self.WLz_x_uni)  # word size
            dist_vector += self.bz_x
            dist_vector = tf.nn.relu(dist_vector)
            dist_vector = tf.matmul(dist_vector, self.WLz_x_uni2)  # word size
            dist_vector += self.bz_x2
            # dist_vector = tf.nn.relu(dist_vector)
        mean, std = tf.split(dist_vector, num_or_size_splits=2, axis=-1)
        std = tf.nn.softplus(std)
        dist_vector = tf.concat([mean, std], axis=-1)
        return dist_vector


    def _step_op_vdecoder(self, time, step, step_gt, memory_state,
                         controller_state=None, controller_hiddens=None):
        """
        performs a step operation on the input step data, assume input is the true y
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6]  # read values from memory

        alphas = None
        controller=self.controller2
        if not self.use_emb_decoder:
            step2 = tf.reshape(step, [-1, self.output_size])
        else:
            step2 = step

        if self.attend_dim > 0:
            values = utility.pack_into_tensor(controller_hiddens, axis=1)
            value_size = self.hidden_controller_dim
            if self.use_mem:
                value_size = self.word_size
            # values = controller_hiddens.gather(tf.range(self.sequence_length))
            encoder_outputs = \
                tf.reshape(values, [self.batch_size, -1, value_size])  # bs x Lin x h
            v = tf.reshape(tf.matmul(tf.reshape(encoder_outputs, [-1, value_size]), self.U_a),
                           [self.batch_size, -1, self.attend_dim])

            if self.use_mem:
                v += tf.reshape(
                    tf.matmul(tf.reshape(last_read_vectors, [-1, self.read_heads_decode * self.word_size]), self.V_a),
                    [self.batch_size, 1, self.attend_dim])

            if self.nlayer > 1:
                try:
                    ns = controller_state[-1][-1]
                    print('multilayer state include c and h')
                except:
                    ns = controller_state[-1]
                    print('multilayer state include only h')
            else:
                ns = controller_state[-1]
                print('single layer')
            print(ns)
            v += tf.reshape(
                tf.matmul(tf.reshape(ns, [-1, self.hidden_controller_dim]), self.W_a),
                [self.batch_size, 1, self.attend_dim])  # bs.Lin x h_att
            print('state include only h')

            v = tf.reshape(tf.tanh(v), [-1, self.attend_dim])
            eijs = tf.matmul(v, tf.expand_dims(self.v_a, 1))  # bs.Lin x 1
            eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
            alphas = tf.nn.softmax(eijs)


            if  not self.use_mem:
                att = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
                att = tf.reshape(att, [self.batch_size, self.hidden_controller_dim])  # bs x h
                step2 = tf.concat([step2, att], axis=-1)  # bs x (decoder_input_size + h)


        mixture_w =   tf.matmul(controller_state[-1][1], self.Wmixw)
        mixture_w = tf.nn.softmax(mixture_w, dim=-1)  # [self.batch_size, self.read_heads_decode]
        if self.use_mem:
            dist2 = self.get_the_prior_dist(last_read_vectors, self.word_size)
        else:
            dist2 = self.get_the_pior_dist_uni(controller_state[-1][1])

        if self.use_mem:
            dist1 = self.get_the_posterior_dist(tf.concat([tf.reduce_sum(tf.tile(tf.expand_dims(mixture_w,1),[1,self.word_size,1])*\
                                                                     last_read_vectors,axis=-1), step_gt], axis=-1))
        else:
            dist1 = self.get_the_posterior_dist_uni(tf.concat([controller_state[-1][1],step_gt], axis=-1))

        if self.use_mem:
            kl = self.KLmixgauss_log(dist1, dist2, mixture_w, self.word_size//2)
        else:
            kl = self.KL2gauss_log(dist1, dist2, self.word_size//2)

        def sample_posterior():
            return self.sample_the_uni(dist1)

        def sample_prior():
            if self.use_mem:
                return self.sample_the_mixture_hard(dist2, mixture_w)
            else:
                return self.sample_the_uni(dist2)

        z = tf.cond(self.testing_phase, sample_prior, sample_posterior)

        r = tf.matmul(z, self.Wz2rnn)  # word size
        # r = tf.nn.relu(r)

        pre_output, interface, nn_state = controller.process_input(step2, r, controller_state)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first
        if self.write_protect:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector \
                =memory_state[1], memory_state[4], memory_state[0], memory_state[3], memory_state[2]

        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading
        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )


        fout = pre_output



        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix,  # 0

            # neccesary for next step to compute memory stuffs
            usage_vector,  # 1
            precedence_vector,  # 2
            link_matrix,  # 3
            write_weighting,  # 4
            read_weightings,  # 5
            read_vectors,  # 6

            # the final output of dnc
            fout,  # 7

            # the values public info to outside
            interface['read_modes'],  # 8
            interface['allocation_gate'],  # 9
            interface['write_gate'],  # 10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1),  # 11
            kl, #12 : the KL lost at current time step
            z, #13 :sampled latent code
            dist1, #14  : posterior
            dist2, #15 : prior
            mixture_w, #16: mixture weight
            last_read_vectors, #17: last reads
        ]


    def _loop_body_vdecoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                           read_weightings, write_weightings, usage_vectors, controller_state,
                           outputs_cache, controller_hiddens,
                           encoder_write_weightings,
                           encoder_controller_hiddens,
                           kl_losses, zs, dist1s, dist2s, mixturews, last_reads):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input
        if self.decoder_mode:
            def fn1():
                return tf.zeros([self.batch_size, self.output_size])
            def fn2():
                def fn2_1():
                    return self.target_output[:, time - 1, :]

                def fn2_2():
                    inds = tf.argmax(outputs_cache.read(time - 1), axis=-1)
                    return tf.one_hot(inds, depth=self.output_size)

                if self.use_teacher:
                    return tf.cond(self.teacher_force[time - 1], fn2_1, fn2_2)
                else:
                    return  fn2_2()

            feed_value = tf.cond(time>0,fn2,fn1)


            if not self.use_emb_decoder:
                r = tf.reshape(feed_value, [self.batch_size, self.input_decoder_size])
                step_input = r
            elif self.dual_emb:
                step_input = tf.matmul(feed_value, self.W_emb_decoder)
            else:
                step_input = tf.matmul(feed_value, self.W_emb_encoder)

        else:
            if self.use_emb_decoder:
                if self.dual_emb:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_decoder)
                else:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_encoder)
            else:
                step_input = self.unpacked_input_decoder_data.read(time)

        if self.gt_type=='bow':
            """bag of word encoding of given data"""
            feed_value_gt = tf.reduce_sum(self.target_output[:,:time+1,:], axis=1)

            if self.use_emb_decoder:
                if self.dual_emb:
                    step_gt = tf.matmul(feed_value_gt, self.W_emb_decoder)
                else:
                    step_gt = tf.matmul(feed_value_gt, self.W_emb_encoder)
            else:
                step_gt = feed_value_gt
        else:
            """rnn encoding of given data"""
            step_gt = self.encode_gt_rnn(time+1)

        #weaken posterior
        step_gt = tf.nn.dropout(step_gt, keep_prob=0.7)
        # compute one step of controller
        output_list = self._step_op_vdecoder(time, step_input, step_gt, memory_state, controller_state)

        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11]  # state hidden  values

        if self.nlayer > 1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')



        outputs = outputs.write(time, output_list[7])  # new output is updated
        outputs_cache = outputs_cache.write(time, output_list[7])  # new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])
        kl_losses = kl_losses.write(time, output_list[12])
        zs = zs.write(time, output_list[13])
        dist1s = dist1s.write(time, output_list[14])
        dist2s = dist2s.write(time, output_list[15])
        mixturews = mixturews.write(time, output_list[16])
        last_reads = last_reads.write(time, output_list[17])

        # all variables have been updated should be return for next step reference
        return (
            time + 1,  # 0
            new_memory_state,  # 1
            outputs,  # 2
            free_gates, allocation_gates, write_gates,  # 3 4 5
            read_weightings, write_weightings, usage_vectors,  # 6 7 8
            new_controller_state,  # 9
            outputs_cache,  # 10
            controller_hiddens,  # 11
            encoder_write_weightings, #12
            encoder_controller_hiddens, #13
            kl_losses, #14
            zs, #15
            dist1s, #16
            dist2s, #17
            mixturews, #18
            last_reads, #19
        )

    def _step_op_decoder(self, time, step, memory_state,
                         controller_state=None, controller_hiddens=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6]  # read values from memory
        pre_output, interface, nn_state = None, None, None

        if self.dual_controller:
            controller=self.controller2
        else:
            controller=self.controller
        alphas = None
        # compute outputs from controller
        if controller.has_recurrent_nn:
            if not self.use_emb_decoder:
                step2 = tf.reshape(step, [-1, self.output_size])
            else:
                step2 = step
            # attention

            if self.attend_dim>0:
                values = utility.pack_into_tensor(controller_hiddens,axis=1)
                value_size = self.hidden_controller_dim
                if self.use_mem:
                    value_size = self.word_size
                # values = controller_hiddens.gather(tf.range(self.sequence_length))
                encoder_outputs = \
                    tf.reshape(values, [self.batch_size, -1, value_size])  # bs x Lin x h
                v = tf.reshape(tf.matmul(tf.reshape(encoder_outputs, [-1, value_size]), self.U_a),
                               [self.batch_size, -1, self.attend_dim])


                if self.use_mem:
                    v+= tf.reshape(
                        tf.matmul(tf.reshape(last_read_vectors, [-1, self.read_heads_decode*self.word_size]), self.V_a),
                        [self.batch_size, 1, self.attend_dim])

                if self.nlayer>1:
                    try:
                        ns=controller_state[-1][-1]
                        print('multilayer state include c and h')
                    except:
                        ns = controller_state[-1]
                        print('multilayer state include only h')
                else:
                    ns = controller_state[-1]
                    print('single layer')
                print(ns)
                v += tf.reshape(
                    tf.matmul(tf.reshape(ns, [-1, self.hidden_controller_dim]), self.W_a),
                    [self.batch_size, 1, self.attend_dim])  # bs.Lin x h_att
                print('state include only h')

                v = tf.reshape(tf.tanh(v), [-1, self.attend_dim])
                eijs = tf.matmul(v, tf.expand_dims(self.v_a, 1))  # bs.Lin x 1
                eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
                alphas = tf.nn.softmax(eijs)

                if not self.use_mem:
                    att = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
                    att = tf.reshape(att, [self.batch_size, self.hidden_controller_dim])  # bs x h
                    step2 = tf.concat([step2, att], axis=-1)  # bs x (decoder_input_size + h)

            pre_output, interface, nn_state = controller.process_input(step2, last_read_vectors, controller_state)

        else:
            pre_output, interface = controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first
        if self.write_protect:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector \
                =memory_state[1], memory_state[4], memory_state[0], memory_state[3], memory_state[2]

        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading

        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        fout = controller.final_output(pre_output, read_vectors) # bs x output_size



        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix,  # 0

            # neccesary for next step to compute memory stuffs
            usage_vector,  # 1
            precedence_vector,  # 2
            link_matrix,  # 3
            write_weighting,  # 4
            read_weightings,  # 5
            read_vectors,  # 6

            # the final output of dnc
            fout,  # 7

            # the values public info to outside
            interface['read_modes'],  # 8
            interface['allocation_gate'],  # 9
            interface['write_gate'],  # 10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1),  # 11
        ]


    def _loop_body_decoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                           read_weightings, write_weightings, usage_vectors, controller_state,
                           outputs_cache, controller_hiddens,
                           encoder_write_weightings, encoder_controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input
        if self.decoder_mode:
            def fn1():
                return tf.zeros([self.batch_size, self.output_size])
            def fn2():
                def fn2_1():
                    return self.target_output[:, time - 1, :]

                def fn2_2():
                    inds = tf.argmax(outputs_cache.read(time - 1), axis=-1)
                    return tf.one_hot(inds, depth=self.output_size)

                if self.use_teacher:
                    return tf.cond(self.teacher_force[time - 1], fn2_1, fn2_2)
                else:
                    return  fn2_2()

            feed_value = tf.cond(time>0,fn2,fn1)


            if not self.use_emb_decoder:
                r = tf.reshape(feed_value, [self.batch_size, self.input_decoder_size])
                step_input = r
            elif self.dual_emb:
                step_input = tf.matmul(feed_value, self.W_emb_decoder)
            else:
                step_input = tf.matmul(feed_value, self.W_emb_encoder)

        else:
            if self.use_emb_decoder:
                if self.dual_emb:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_decoder)
                else:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_encoder)
            else:
                step_input = self.unpacked_input_decoder_data.read(time)
                print(step_input.shape)
                print('ssss')

        # compute one step of controller
        if not self.use_mem and self.attend_dim > 0:
            print('normal attention or mix pointer mode without memory')
            output_list = self._step_op_decoder(time, step_input, memory_state, controller_state, encoder_controller_hiddens)
        elif self.use_mem and self.attend_dim > 0:
            print('attention and mix pointer mode with memory')
            output_list = self._step_op_decoder(time, step_input, memory_state, controller_state, encoder_write_weightings)
        else:
            output_list = self._step_op_decoder(time, step_input, memory_state, controller_state)
            # update memory parameters
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11] # state hidden  values

        if self.nlayer>1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')
        outputs = outputs.write(time, output_list[7])  # new output is updated
        outputs_cache = outputs_cache.write(time, output_list[7])  # new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1,  # 0
            new_memory_state,  # 1
            outputs,  # 2
            free_gates, allocation_gates, write_gates,  # 3 4 5
            read_weightings, write_weightings, usage_vectors,  # 6 7 8
            new_controller_state,  # 9
            outputs_cache,  # 10
            controller_hiddens,  # 11
            encoder_write_weightings, #12
            encoder_controller_hiddens, #13
        )

    def _step_op_encoder(self, time,  step, memory_state, controller_state=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6] # read values from memory
        pre_output, interface, nn_state = None, None, None

        # compute oututs from controller
        if self.controller.has_recurrent_nn:
            # controller state is the rnn cell state pass through each time step
            if not self.use_emb_encoder:
                step2 = tf.reshape(step, [-1, self.input_encoder_size])
                pre_output, interface, nn_state=  self.controller.process_input(step2, last_read_vectors, controller_state)
            else:
                pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first




        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector= \
            self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading


        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )
        fout=None
        if self.use_encoder_output:
            fout = self.controller.final_output(pre_output, read_vectors)


        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix, #0

            # neccesary for next step to compute memory stuffs
            usage_vector, #1
            precedence_vector, #2
            link_matrix, #3
            write_weighting, #4
            read_weightings, #5
            read_vectors, #6

            # the final output of dnc
            fout, #7

            # the values public info to outside
            interface['read_modes'], #8
            interface['allocation_gate'], #9
            interface['write_vector'], #10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1), #11
        ]

    def _loop_body_encoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state,
                   outputs_cache, controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input

        if self.use_emb_encoder:
            step_input = tf.matmul(self.unpacked_input_encoder_data.read(time), self.W_emb_encoder)
        else:
            step_input = self.unpacked_input_encoder_data.read(time)


        # compute one step of controller
        output_list = self._step_op_encoder(time, step_input, memory_state, controller_state)
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11] #state  hidden values

        if self.nlayer>1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')

        if self.use_encoder_output:
            outputs = outputs.write(time, output_list[7])# new output is updated
            outputs_cache = outputs_cache.write(time, output_list[7])# new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[10])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1, #0
            new_memory_state, #1
            outputs, #2
            free_gates,allocation_gates, write_gates, #3 4 5
            read_weightings, write_weightings, usage_vectors, #6 7 8
            new_controller_state, #9
            outputs_cache,  #10
            controller_hiddens, #11
        )

    def _loop_body_vencoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                           read_weightings, write_weightings, usage_vectors, controller_state,
                           outputs_cache, controller_hiddens, input_data):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input

        if self.use_emb_encoder:
            step_input = tf.matmul(input_data.read(time), self.W_emb_encoder)
        else:
            step_input = input_data.read(time)

        # compute one step of controller

        output_list = self._step_op_encoder(time, step_input, memory_state, controller_state)
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11]  # state  hidden values

        if self.nlayer>1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')

        if self.use_encoder_output:
            outputs = outputs.write(time, output_list[7])# new output is updated
            outputs_cache = outputs_cache.write(time, output_list[7])# new output is updated
        # collecting memory view for the current step
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[10])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1,  # 0
            new_memory_state,  # 1
            outputs,  # 2
            free_gates, allocation_gates, write_gates,  # 3 4 5
            read_weightings, write_weightings, usage_vectors,  # 6 7 8
            new_controller_state,  # 9
            outputs_cache,  # 10
            controller_hiddens,  # 11
            input_data, #12
        )

    def encode_gt_rnn(self, gtime):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_controller_dim)
        decoder_state = rnn_cell.zero_state(self.batch_size, tf.float32)
        time = tf.constant(0, dtype=tf.int32)
        doutputs = tf.TensorArray(tf.float32, self.decode_length)

        def gt_body_loop(_time, _decoder_state, _outputs):



            X = self.target_output[:,_time,:]

            X = tf.reshape(X, [self.batch_size, self.output_size])
            if self.dual_emb:
                X = tf.matmul(X, self.W_emb_decoder)
            else:
                X = tf.matmul(X, self.W_emb_encoder)

            _, nns = rnn_cell(X, _decoder_state)


            return (
                _time + 1,  # 0
                nns,  # 1
                _outputs,  # 3
            )


        decoder_results = tf.while_loop(
            cond=lambda time, *_: time < gtime,
            body=gt_body_loop,
            loop_vars=(
                time, decoder_state, doutputs
            ),  # do not need to provide intial values, the initial value lies in the variables themselves
            parallel_iterations=1,
            swap_memory=True
        )

        rnn_feature=tf.concat(decoder_results[1], axis=-1)
        return rnn_feature



    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        # make dynamic time step length tensor
        self.unpacked_input_encoder_data = utility.unpack_into_tensorarray(self.input_encoder, 1, self.sequence_length)

        # want to store all time step values of these variables
        eoutputs = tf.TensorArray(tf.float32, self.sequence_length)
        eoutputs_cache = tf.TensorArray(tf.float32, self.sequence_length)
        efree_gates = tf.TensorArray(tf.float32, self.sequence_length)
        eallocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        ewrite_gates = tf.TensorArray(tf.float32, self.sequence_length)
        eread_weightings = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        ewrite_weightings = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        eusage_vectors = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        econtroller_hiddens = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)

        # make dynamic time step length tensor
        self.unpacked_input_decoder_data = utility.unpack_into_tensorarray(self.input_decoder, 1, self.decode_length)
        self.unpacked_target_data = utility.unpack_into_tensorarray(self.target_output, 1, self.decode_length)

        # want to store all time step values of these variables
        doutputs = tf.TensorArray(tf.float32, self.decode_length)
        doutputs_cache = tf.TensorArray(tf.float32, self.decode_length)
        dfree_gates = tf.TensorArray(tf.float32, self.decode_length)
        dallocation_gates = tf.TensorArray(tf.float32, self.decode_length)
        dwrite_gates = tf.TensorArray(tf.float32, self.decode_length)
        dread_weightings = tf.TensorArray(tf.float32, self.decode_length)
        dwrite_weightings = tf.TensorArray(tf.float32, self.decode_length, clear_after_read=False)
        dusage_vectors = tf.TensorArray(tf.float32, self.decode_length)
        dcontroller_hiddens = tf.TensorArray(tf.float32, self.decode_length, clear_after_read=False)
        dkl = tf.TensorArray(tf.float32, self.decode_length)
        zs = tf.TensorArray(tf.float32, self.decode_length)
        dist1s = tf.TensorArray(tf.float32, self.decode_length)
        dist2s = tf.TensorArray(tf.float32, self.decode_length)
        mixturews = tf.TensorArray(tf.float32, self.decode_length)
        last_reads = tf.TensorArray(tf.float32, self.decode_length)

        # inital state for RNN controller
        controller_state = self.controller.zero_state()
        print(controller_state)
        memory_state = self.memory.init_memory()


        # final_results = None
        with tf.variable_scope("sequence_encoder_loop"):
            time = tf.constant(0, dtype=tf.int32)

            # use while instead of scan --> suitable with dynamic time step
            encoder_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body_encoder,
                loop_vars=(
                    time, memory_state, eoutputs,
                    efree_gates, eallocation_gates, ewrite_gates,
                    eread_weightings, ewrite_weightings,
                    eusage_vectors, controller_state,
                    eoutputs_cache, econtroller_hiddens
                ), # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=1,
                swap_memory=True
            )

        single_kl = 0
        memory_state2 = self.memory.init_memory(self.read_heads_decode)
        if self.read_heads_decode!=self.read_heads:
            encoder_results_state=(encoder_results[1][0],encoder_results[1][1],encoder_results[1][2],
                                encoder_results[1][3],encoder_results[1][4], memory_state2[5],memory_state2[6])
        else:
            encoder_results_state=encoder_results[1]
        with tf.variable_scope("sequence_vdecoder_loop"):
            time = tf.constant(0, dtype=tf.int32)
            nstate = controller_state
            if self.pass_encoder_state:
                nstate = encoder_results[9]


            if self.single_KL:
                def cal_h_y():
                    time = tf.constant(0, dtype=tf.int32)

                    # use while instead of scan --> suitable with dynamic time step
                    encoder_results_y = tf.while_loop(
                        cond=lambda time, *_: time < self.decode_length,
                        body=self._loop_body_vencoder,
                        loop_vars=(
                            time, memory_state, eoutputs,
                            efree_gates, eallocation_gates, ewrite_gates,
                            eread_weightings, ewrite_weightings,
                            eusage_vectors, controller_state,
                            eoutputs_cache, econtroller_hiddens, self.unpacked_target_data
                        ),  # do not need to provide intial values, the initial value lies in the variables themselves
                        parallel_iterations=1,
                        swap_memory=True
                    )
                    print(encoder_results[9])
                    return encoder_results_y[9]

                def cal_h_wt_y():
                    print(self.controller.zero_state())
                    return self.controller.zero_state()

                nstate_y = tf.cond(self.testing_phase, cal_h_wt_y, cal_h_y)
                if self.nlayer==1:
                    nstate=[nstate]
                    nstate_y=[nstate_y]


                def compute_single_z_y():
                    newnstate = []
                    for ns, ns_y in zip(nstate, nstate_y):
                        nh = tf.concat([ns[0]+ns_y[0],ns[1]+ns_y[1]],axis=-1)
                        nh = tf.matmul(tf.reshape(nh, [self.batch_size, -1]), self.W_recog)
                        dist = self.get_the_pior_dist_uni(tf.tanh(nh))
                        z = self.sample_the_uni(dist)
                        newnstate.append(LSTMStateTuple(ns[0],z))
                    return newnstate, dist

                def compute_single_z():
                    newnstate = []
                    wm=None
                    for ns in nstate:
                        nh0 = tf.concat([ns[0], ns[1]], axis=-1)
                        nh = tf.matmul(tf.reshape(nh0, [self.batch_size, -1]), self.W_pior)
                        if self.read_heads_decode==1:
                            dist = self.get_the_pior_dist_uni(nh)
                            z = self.sample_the_uni(dist)
                        else:
                            dist = self.get_the_prior_dist(
                                tf.reshape(nh,[self.batch_size,self.hidden_controller_dim*2,self.read_heads_decode]),
                                self.hidden_controller_dim)
                            dist = tf.sigmoid(dist)
                            wm = tf.matmul(tf.reshape(nh0, [self.batch_size, -1]), self.W_modew)
                            wm=tf.nn.softmax(wm, dim=-1)
                            z = self.sample_the_mixture(dist, wm)
                        newnstate.append(LSTMStateTuple(ns[0], z))
                    return newnstate, dist, wm

                newnstate_y, dist_y = compute_single_z_y()
                newnstate_x, dist_x, wm = compute_single_z()

                def ns_y():
                    return newnstate_y

                def ns_x():
                    return newnstate_x

                newnstate = tf.cond(self.testing_phase, ns_x, ns_y)


                if self.read_heads_decode==1:
                    single_kl = self.KL2gauss_log(dist_y, dist_x, self.hidden_controller_dim)
                else:
                    single_kl = self.KLmixgauss_log(dist_y, dist_x, wm, self.hidden_controller_dim)
                if self.nlayer==1:
                    newnstate=newnstate[0]
                nstate=tuple(newnstate)


                final_results = tf.while_loop(
                    cond=lambda time, *_: time < self.decode_length,
                    body=self._loop_body_decoder,
                    loop_vars=(
                        time, encoder_results_state, doutputs,
                        dfree_gates, dallocation_gates, dwrite_gates,
                        dread_weightings, dwrite_weightings,
                        dusage_vectors, nstate,
                        doutputs_cache, dcontroller_hiddens,
                        encoder_results[7], encoder_results[11]
                    ),  # do not need to provide intial values, the initial value lies in the variables themselves
                    parallel_iterations=1,
                    swap_memory=True
                )
            else:
                # use while instead of scan --> suitable with dynamic time step
                final_results = tf.while_loop(
                    cond=lambda time, *_: time < self.decode_length,
                    body=self._loop_body_vdecoder,
                    loop_vars=(
                        time, encoder_results_state, doutputs,
                        dfree_gates, dallocation_gates, dwrite_gates,
                        dread_weightings, dwrite_weightings,
                        dusage_vectors, nstate,
                        doutputs_cache, dcontroller_hiddens,
                        encoder_results[7], encoder_results[11],
                        dkl, zs, dist1s, dist2s, mixturews, last_reads
                    ),  # do not need to provide intial values, the initial value lies in the variables themselves
                    parallel_iterations=1,
                    swap_memory=True
                )



        dependencies = []
        if self.controller.has_recurrent_nn:
            # tensor array of pair of hidden and state values of rnn
            dependencies.append(self.controller.update_state(final_results[9]))

        with tf.control_dependencies(dependencies):
            # convert output tensor array to normal tensor
            self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
            if self.single_KL:
                self.packed_kl_losses = single_kl
            else:
                self.packed_kl_losses = utility.pack_into_tensor(final_results[14], axis=1)
            self.packed_memory_view_encoder = {
                'free_gates': utility.pack_into_tensor(encoder_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(encoder_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(encoder_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(encoder_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(encoder_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(encoder_results[8], axis=1),
                'final_controller_ch': encoder_results[9],
            }
            self.packed_memory_view_decoder = {
                'last_reads': utility.pack_into_tensor(final_results[19], axis=1),
                'free_gates': utility.pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(final_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(final_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(final_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(final_results[8], axis=1),
                'final_controller_ch':final_results[9],
                'zs':utility.pack_into_tensor(final_results[15], axis=1),
                'dist1s': utility.pack_into_tensor(final_results[16], axis=1),
                'dist2s': utility.pack_into_tensor(final_results[17], axis=1),
                'mixturews': utility.pack_into_tensor(final_results[18], axis=1)
            }

    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view
        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view_encoder, self.packed_memory_view_decoder

    def build_vloss_function_mask(self, optimizer=None, clip_s=10, total_epoch=1000000):

        # train_arg.add_argument('--lr_start', type=float, default=0.001, help='')
        # train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='')
        # train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='')
        # train_arg.add_argument('--max_grad_norm', type=float, default=2.0, help='')

        print('build loss mask....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _, _ = self.get_outputs()
        prob = tf.nn.softmax(output, dim=-1)

        score = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_output,
            logits=output, dim=-1)
        score_flatten = tf.reshape(score, [-1])
        mask_flatten = tf.reshape(self.mask, [-1])
        mask_score = tf.boolean_mask(score_flatten, mask_flatten)
        if self.KL_anneal:
            alpha = tf.minimum(self.epochs/total_epoch,1.0)
        else:
            alpha=tf.constant(1.0)
        loss_rec = tf.reduce_mean(mask_score)
        loss_kl = tf.reduce_mean(self.packed_kl_losses)
        loss = alpha*1.0*loss_kl+loss_rec



        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                if isinstance(clip_s, list):
                    gradients[i] = (tf.clip_by_value(grad, clip_s[0], clip_s[1]), var)
                else:
                    gradients[i] = (tf.clip_by_norm(grad, clip_s), var)

        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients, loss_rec, loss_kl, alpha

    def print_config(self):
        return '{}.{}mem_{}dec_{}dua_{}wrp_{}wsz_{}msz_{}tea_{}att_{}hid_{}nread_{}nlayer'.\
            format(self.name, self.use_mem,
                   self.decoder_mode,
                   self.dual_controller,
                   self.write_protect,
                   self.words_num,
                   self.word_size,
                   self.use_teacher,
                   self.attend_dim,
                   self.hidden_controller_dim,
                   self.read_heads_decode,
                   self.nlayer)

    def assign_pretrain_emb_encoder(self, sess, lookup_mat):
        assign_op_W_emb_encoder = self.W_emb_encoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_encoder])

    def assign_pretrain_emb_decoder(self, sess, lookup_mat):
        assign_op_W_emb_decoder = self.W_emb_decoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_decoder])

    @staticmethod
    def get_size_model():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    @staticmethod
    def save(session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint
        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.global_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))


    @staticmethod
    def restore(session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.global_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))

    @staticmethod
    def get_bool_rand(size_seq, prob_true=0.1):
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_incremental(size_seq, prob_true_min=0, prob_true_max=0.25):
        ret = []
        for i in range(size_seq):
            prob_true=(prob_true_max-prob_true_min)/size_seq*i
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_curriculum(size_seq, epoch, k=0.99, type='exp'):
        if type=='exp':
            prob_true = k**epoch
        elif type=='sig':
            prob_true = k / (k + np.exp(epoch / k))
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)