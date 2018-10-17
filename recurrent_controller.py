import tensorflow as tf
from controller import BaseController



class StatelessRecurrentController(BaseController):
    def network_vars(self):
        print('--define core rnn stateless controller variables--')

        cell = None

        if self.cell_type == "nlstm":
            cell= tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_dim, layer_norm = self.batch_norm,
                                                    dropout_keep_prob=self.drop_out_keep)

            self.cell_weight_name="layer_norm_basic_lstm_cell/kernel"
        else:
            if self.cell_type == "lstm":
                cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            if self.cell_type == "igru":
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.keras.initializers.Identity(gain=1.0))
            elif self.cell_type == "gru":
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            elif self.cell_type == "rnn":
                cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)
            if not isinstance(self.drop_out_keep, int):
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                     input_keep_prob=self.drop_out_keep)

        if self.nlayer==1:
            print('1 layer')
            self.controller_cell = cell
        else:
            print('{} layers'.format(self.nlayer))
            if self.cell_type == "nlstm":
                self.controller_cell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_dim, layer_norm = self.batch_norm,
                                                                                                          dropout_keep_prob=self.drop_out_keep) for _ in range(self.nlayer)])
            elif self.cell_type == "lstm":
                layers=[]
                for _ in range(self.nlayer):
                    cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
                    if not isinstance(self.drop_out_keep, int):
                        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=self.drop_out_keep)
                    layers.append(cell)
                self.controller_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
            elif self.cell_type == "gru":
                layers=[]
                for _ in range(self.nlayer):
                    cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
                    if not isinstance(self.drop_out_keep, int):
                        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=self.drop_out_keep)
                    layers.append(cell)
                self.controller_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
            elif self.cell_type == "rnn":
                layers = []
                for _ in range(self.nlayer):
                    cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)
                    if not isinstance(self.drop_out_keep, int):
                        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                             input_keep_prob=self.drop_out_keep)
                    layers.append(cell)
                self.controller_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

        print("controller cell")
        print(self.controller_cell)

        self.state = self.controller_cell.zero_state(self.batch_size, tf.float32)





    def network_op(self, X, state):
        print('--operation rnn stateless controller variables--')
        X = tf.convert_to_tensor(X)
        return self.controller_cell(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()

    def zero_state(self):
        return self.controller_cell.zero_state(self.batch_size, tf.float32)