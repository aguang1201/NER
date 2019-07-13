from kashgari.corpus import CoNLL2003Corpus
from kashgari.tasks.seq_labeling import BLSTMCRFModel
import tensorflow as tf
from configparser import ConfigParser
from clr_callback import *


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
        f.write(f"The evaluate report is :\n {str(report_evaluate)}\n")

if __name__ == "__main__":
    set_sess_cfg()
    main()

'''
           precision    recall  f1-score   support

     MISC     0.7314    0.6838    0.7068       661
      ORG     0.7743    0.7393    0.7564      1592
      LOC     0.8484    0.8758    0.8619      1578
      PER     0.8436    0.8548    0.8492      1495

micro avg     0.8120    0.8053    0.8086      5326
macro avg     0.8104    0.8053    0.8075      5326
'''