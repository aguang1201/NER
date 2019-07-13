from kashgari.corpus import CoNLL2003Corpus
from kashgari.tasks.seq_labeling import BLSTMCRFModel
import tensorflow as tf
from configparser import ConfigParser
from clr_callback import *
import logging


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
    model_fold = cp["TEST"].get("model_fold")
    output_dir = os.path.join('experiments', model_fold)

    model_path = os.path.join(output_dir, 'model')
    model = BLSTMCRFModel.load_model(model_path)
    sentence = 'China and the United States are about the same size'
    sentence_list = sentence.split()
    result = model.predict(sentence_list)
    result_dict = model.predict(sentence_list, output_dict=True)
    print(f'the sentence is {sentence}')
    print(f'the result is {result}')
    print(f'the result of dict is {result_dict}')
    logging.info('test predict: {} -> {}'.format(sentence_list, result))

    with open(os.path.join(output_dir, 'result_predict.log'), 'w') as f:
        f.write(f"The predict result is : {str(result)}\n")

if __name__ == "__main__":
    set_sess_cfg()
    main()

'''
the sentence is China and the United States are about the same size
the result is ['B-LOC', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O']
the result of dict is {'words': ['China', 'and', 'the', 'United', 'States', 'are', 'about', 'the', 'same', 'size'], 'entities': [{'text': 'China', 'type': 'LOC', 'beginOffset': 0, 'endOffset': 1}, {'text': 'United States', 'type': 'LOC', 'beginOffset': 3, 'endOffset': 5}]}
'''