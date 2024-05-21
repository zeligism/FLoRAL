import tensorflow as tf


class VocabLookup(tf.keras.layers.Layer):
    """
    tf's resource tensors are not serializable, so we either have to init tokenizer
    lazily when getting client loader (so that lookup table is loaded within the ray actor,
    which is really slow), or we can just try to figure out how to serialize this mf
    (thank you, Nicholas Leonard: https://stackoverflow.com/a/58507856)
    """
    def __init__(self, vocab, num_oov_buckets=1, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.num_oov_buckets = num_oov_buckets

    def build(self, input_shape):
        table_values = tf.range(len(self.vocab), dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(self.vocab, table_values),
            self.num_oov_buckets)
        self.built = True

    def call(self, keys):
        return self.table.lookup(keys)

    def get_config(self):
        return {'vocab': self.vocab, 'num_oov_buckets': self.num_oov_buckets}
