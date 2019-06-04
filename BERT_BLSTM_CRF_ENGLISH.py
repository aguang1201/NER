from kashgari.corpus import CoNLL2003Corpus
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
import tensorflow as tf
from keras import backend as K


def set_sess_cfg():
    config_sess = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config_sess.gpu_options.allow_growth = True
    sess = tf.Session(config=config_sess)
    K.set_session(sess)

set_sess_cfg()
train_x, train_y = CoNLL2003Corpus.get_sequence_tagging_data('train')
validate_x, validate_y = CoNLL2003Corpus.get_sequence_tagging_data('validate')
test_x, test_y = CoNLL2003Corpus.get_sequence_tagging_data('test')
print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
print(f"test data count: {len(test_x)}")

#'bert-base-uncased'
embedding = BERTEmbedding('bert-large-cased', 30)
# 还可以选择 `BLSTMModel` 和 `CNNLSTMModel`
model = BLSTMCRFModel(embedding)
# model.build_model(train_x, train_y)
# model.build_multi_gpu_model(gpus=2)
model.fit(train_x,
          train_y,
          x_validate=validate_x,
          y_validate=validate_y,
          epochs=200,
          batch_size=500)
