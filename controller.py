import tensorflow as tf
import numpy as np

class BaseController:

    def __init__(self, input_size, output_size, memory_read_heads, memory_word_size, batch_size=1,
                 use_mem=True, hidden_dim=256, is_two_mem=0, cell_type="nlstm",
                 drop_out_keep=1, batch_norm=False, vae_mode=False, nlayer=1, clip_output=0):
        """
        constructs a controller as described in the DNC paper:
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        ----------
        input_size: int
            the size of the data input vector
        output_size: int
            the size of the data output vector
        memory_read_heads: int
            the number of read heads in the associated external memory
        memory_word_size: int
            the size of the word in the associated external memory
        batch_size: int
            the size of the input data batch [optional, usually set by the DNC object]
        """
        self.use_mem = use_mem
        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = memory_read_heads # in dnc there are many read head but one write head
        self.word_size = memory_word_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.drop_out_keep = drop_out_keep
        self.batch_norm= batch_norm
        self.vae_mode = vae_mode
        self.nlayer = nlayer
        self.cell_type = cell_type
        self.clip_output = clip_output
        # indicates if the internal neural network is recurrent
        # by the existence of recurrent_update and get_state methods
        # subclass should implement these methods if it is rnn based controller
        has_recurrent_update = callable(getattr(self, 'update_state', None))
        has_get_state = callable(getattr(self, 'get_state', None))
        self.has_recurrent_nn =  has_recurrent_update and has_get_state

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory
        if use_mem or self.vae_mode:
            if is_two_mem>0:
                self.nn_input_size = self.word_size * self.read_heads*2 + self.input_size
            elif self.vae_mode:
                self.nn_input_size = self.word_size + self.input_size
            else:
                self.nn_input_size = self.word_size * self.read_heads + self.input_size
        else:
            self.nn_input_size = self.input_size

        self.interface_vector_size = self.word_size * self.read_heads #R read keys
        self.interface_vector_size += 3 * self.word_size #1 write key, 1 erase, 1 content
        self.interface_vector_size += 5 * self.read_heads #R read key strengths, R free gates, 3xR read modes (each mode for each read has 3 values)
        self.interface_vector_size += 3 # 1 write strength, 1 allocation gate, 1 write gate

        self.interface_weights = self.nn_output_weights = self.mem_output_weights = None
        self.is_two_mem = is_two_mem

        # define network vars
        with tf.name_scope("controller"):
            self.network_vars()

            self.nn_output_size = None # not yet defined in the general scope --> output of the controller not of the whole
            with tf.variable_scope("shape_inference"):
                #depend on model type --> seperate variable scope
                self.nn_output_size = self.get_nn_output_size()

            self.initials()

    def initials(self):
        """
        sets the initial values of the controller transformation weights matrices
        this method can be overwritten to use a different initialization scheme
        """
        # defining internal weights of the controller
        if self.is_two_mem==2:
            self.interface_weights = tf.Variable(
                tf.random_normal([self.nn_output_size, self.interface_vector_size*2], stddev=0.1),
                name='interface_weights'
            ) # function to compute interface: i = H x iW
        else:
            self.interface_weights = tf.Variable(
                tf.random_normal([self.nn_output_size, self.interface_vector_size], stddev=0.1),
                name='interface_weights'
            )  # function to compute interface: i = H x iW

        self.nn_output_weights = tf.Variable(
            tf.random_normal([self.nn_output_size, self.output_size], stddev=0.1),
            name='nn_output_weights'
        ) # function to compute output of the whole : v = H x yW
        if self.is_two_mem>0:
            self.mem_output_weights = tf.Variable(
                tf.random_normal([2*self.word_size * self.read_heads, self.output_size], stddev=0.1),
                name='mem_output_weights'
            )

        else:
            # if self.vae_mode:
            #     final_win=self.word_size
            # else:
            final_win = self.word_size * self.read_heads

            self.mem_output_weights = tf.Variable(
                tf.random_normal([final_win, self.output_size],  stddev=0.1),
                name='mem_output_weights'
        ) # function to compute final output of the whole, combine output and read values: y = v + rs x Wr



    def network_vars(self):
        """
        defines the variables needed by the internal neural network
        [the variables should be attributes of the class, i.e. self.*]
        """
        raise NotImplementedError("network_vars is not implemented")


    def network_op(self, X):
        """
        defines the controller's internal neural network operation

        Parameters:
        ----------
        X: Tensor (batch_size, word_size * read_haeds + input_size)
            the input data concatenated with the previously read vectors from memory

        Returns: Tensor (batch_size, nn_output_size)
        """
        raise NotImplementedError("network_op method is not implemented")


    def get_nn_output_size(self):
        """
        retrives the output size of the defined neural network

        Returns: int
            the output's size

        Raises: ValueError
        """

        input_vector =  np.zeros([self.batch_size, self.nn_input_size], dtype=np.float32) #dummy data to get output size

        if self.has_recurrent_nn:
            output_vector,_ = self.network_op(input_vector, self.get_state()) # connacate all steps hidden state vector
        else:
            output_vector = self.network_op(input_vector) # just hidden state vector

        shape = output_vector.get_shape().as_list() # batch x output_size

        if len(shape) > 2:
            raise ValueError("Expected the neural network to output a 1D vector, but got %dD" % (len(shape) - 1))
        else:
            return shape[1]


    def parse_interface_vector(self, interface_vector):
        """
        pasres the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened inetrface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """
        if self.clip_output>0:
            interface_vector = tf.clip_by_value(interface_vector, -self.clip_output, self.clip_output)
        parsed = {}

        r_keys_end = self.word_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.word_size
        erase_end = w_key_end + 1 + self.word_size
        write_end = erase_end + self.word_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.word_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.word_size, 1)
        write_shape = erase_shape = (-1, self.word_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)

        # parsing the vector into its individual components
        parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape) #batch x N x R
        parsed['read_strengths'] = tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_strengths_shape) #batch x R
        parsed['write_key'] = tf.reshape(interface_vector[:, r_strengths_end:w_key_end], w_key_shape) #batch x N x 1 --> share similarity function with read
        parsed['write_strength'] = tf.reshape(interface_vector[:, w_key_end], (-1, 1)) # batch x 1
        parsed['erase_vector'] = tf.reshape(interface_vector[:, w_key_end + 1:erase_end], erase_shape) #batch x N
        parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)# batch x N
        parsed['free_gates'] = tf.reshape(interface_vector[:, write_end:free_end], free_shape)# batch x R
        parsed['allocation_gate'] = tf.expand_dims(interface_vector[:, free_end], 1)# batch x 1
        parsed['write_gate'] = tf.expand_dims(interface_vector[:, free_end + 1], 1)# batch x 1
        parsed['read_modes'] = tf.reshape(interface_vector[:, free_end + 2:], modes_shape)# batch x 3 x R

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + tf.nn.softplus(parsed['read_strengths'])
        parsed['write_strength'] = 1 + tf.nn.softplus(parsed['write_strength'])
        parsed['erase_vector'] = tf.nn.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] = tf.nn.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] = tf.nn.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] = tf.nn.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = tf.nn.softmax(parsed['read_modes'], 1)

        return parsed # dict of tensors

    def process_zero(self):
        pre_output = tf.zeros([self.batch_size, self.output_size],dtype=np.float32)
        interface = tf.zeros([self.batch_size, self.interface_vector_size],dtype=np.float32)
        parsed_interface = self.parse_interface_vector(interface)
        parsed_interface['read_strengths'] *= 0
        parsed_interface['write_strength'] *= 0
        parsed_interface['erase_vector'] *= 0
        parsed_interface['free_gates'] *= 0
        parsed_interface['allocation_gate'] *= 0
        parsed_interface['write_gate'] *= 0
        parsed_interface['read_modes'] *= 0
        if self.has_recurrent_nn:
            return pre_output, parsed_interface, self.lstm_cell.zero_state(self.batch_size, tf.float32)
        else:
            return pre_output, parsed_interface

    def process_input(self, X, last_read_vectors, state=None, compute_interface=True):
        """
        processes input data through the controller network and returns the
        pre-output and interface_vector

        Parameters:
        ----------
        X: Tensor (batch_size, input_size)
            the input data batch
        last_read_vectors: (batch_size, word_size, read_heads)
            the last batch of read vectors from memory
        state: Tuple
            state vectors if the network is recurrent

        Returns: Tuple
            pre-output: Tensor (batch_size, output_size)
            parsed_interface_vector: dict
        """

        flat_read_vectors = tf.reshape(last_read_vectors, (self.batch_size, -1)) #flatten R read vectors: batch x RN
        if self.use_mem or self.vae_mode:
            complete_input = tf.concat([X, flat_read_vectors], 1)#concat input --> read data
        else:
            complete_input = X

        # print('---')
        # print(X.shape)
        # print(flat_read_vectors.shape)
        # print(complete_input.shape)
        if self.has_recurrent_nn:
            nn_output, nn_state = self.network_op(complete_input, state)
            print('recurrent state')
            print(nn_state)
        else:
            nn_output = self.network_op(complete_input)

        pre_output = tf.matmul(nn_output, self.nn_output_weights) #batch x output_dim -->later combine with new read vector

        if compute_interface:
            interface = tf.matmul(nn_output, self.interface_weights) #batch x interface_dim
        else:
            interface = tf.zeros([self.batch_size, self.interface_vector_size])
        if self.is_two_mem==2:
            interface1, interface2 = tf.split(interface, num_or_size_splits=2, axis=-1)
            parsed_interface = (self.parse_interface_vector(interface1),
                                self.parse_interface_vector(interface2))
        else:
            parsed_interface = self.parse_interface_vector(interface) #use to read write into vector

        if self.has_recurrent_nn:
            return pre_output, parsed_interface, nn_state
        else:
            return pre_output, parsed_interface


    def final_output(self, pre_output, new_read_vectors):
        """
        returns the final output by taking rececnt memory changes into account

        Parameters:
        ----------
        pre_output: Tensor (batch_size, output_size)
            the ouput vector from the input processing step
        new_read_vectors: Tensor (batch_size, words_size, read_heads)
            the newly read vectors from the updated memory

        Returns: Tensor (batch_size, output_size)
        """

        flat_read_vectors = tf.reshape(new_read_vectors, (self.batch_size, -1)) # batch_size x flatten
        # final output is combine output from controller and read vectors --> just like concat hidden to read vectors
        # then linear transform

        final_output = pre_output

        if self.use_mem:
            final_output+=tf.matmul(flat_read_vectors, self.mem_output_weights)

        return final_output #same size as pre_output: batch_size x outputdim (classification problem, outputdim=number of labels)


