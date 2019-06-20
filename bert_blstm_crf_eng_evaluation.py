from kashgari.corpus import CoNLL2003Corpus
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
import tensorflow as tf
from configparser import ConfigParser
import datetime
import shutil
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from math import ceil
from clr_callback import *
import matplotlib.pyplot as plt
from callback import SaveMinLoss


def set_sess_cfg():
    config_sess = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config_sess.gpu_options.allow_growth = True
    sess = tf.Session(config=config_sess)
    K.set_session(sess)

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    model_fold = cp["EVALUATION"].get("model_fold")
    output_dir = os.path.join('experiments', model_fold)

    test_x, test_y = CoNLL2003Corpus.get_sequence_tagging_data('test')

    model_path = os.path.join(output_dir, 'model')
    model = BLSTMCRFModel.load_model(model_path)
    report_evaluate = model.evaluate(test_x, test_y, debug_info=True)

    with open(os.path.join(output_dir, 'report_evaluate.log'), 'w') as f:
        f.write(f"The evaluate report is : {str(report_evaluate)}\n")

if __name__ == "__main__":
    set_sess_cfg()
    main()