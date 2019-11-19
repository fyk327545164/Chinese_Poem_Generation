class Config(object):

    def __init__(self):
        
        self.embedding_dim = 128

        self.num_layer = 3

        self.num_lstm_units = 512

        self.dropout_prob = 0.5

        self.batch_size = 64

        self.learning_rate = 0.001

        self.beam_width = 3

        self.num_steps = 50000

        self.eval_inf_step = 200

        self.save_checkpoint_step = 500
