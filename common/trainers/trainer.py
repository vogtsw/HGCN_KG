class Trainer(object):

    """
    Abstraction for training a model on a Dataset.
    """

    def __init__(self, model, embedding=None, train_loader=None, trainer_config=None, train_evaluator=None, test_evaluator=None, dev_evaluator=None):
        self.model = model
        self.embedding = embedding
        self.train_loader = train_loader
        
        if trainer_config is not None:
            self.optimizer = trainer_config.get('optimizer')
            self.batch_size = trainer_config.get('batch_size')
            self.log_interval = trainer_config.get('log_interval')
            self.model_outfile = trainer_config.get('model_outfile')
            self.lr_reduce_factor = trainer_config.get('lr_reduce_factor')
            self.patience = trainer_config.get('patience')
            self.clip_norm = trainer_config.get('clip_norm')
            self.logger = trainer_config.get('logger')
        else:
            self.optimizer = None
            self.batch_size = None
            self.log_interval = None
            self.model_outfile = None
            self.lr_reduce_factor = None
            self.patience = None
            self.clip_norm = None
            self.logger = None

        self.train_evaluator = train_evaluator
        self.test_evaluator = test_evaluator
        self.dev_evaluator = dev_evaluator

    def evaluate(self, evaluator, dataset_name):
        scores, metric_names = evaluator.get_scores()
        if self.logger is not None:
            self.logger.info('Evaluation metrics for {}:'.format(dataset_name))
            self.logger.info('\t'.join([' '] + metric_names))
            self.logger.info('\t'.join([dataset_name] + list(map(str, scores))))
        return scores

    def get_sentence_embeddings(self, batch):
        sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_2).transpose(1, 2)
        return sent1, sent2

    def train_epoch(self, epoch):
        raise NotImplementedError()

    def train(self, epochs):
        raise NotImplementedError()
