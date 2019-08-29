#coding:utf-8
from kashgari.callbacks import EvalCallBack
from kashgari.tasks.classification import BiLSTM_Model, BiGRU_Model, BLSTMModel, BGRUModel
from kashgari.embeddings import BERTEmbedding
# from kashgari.tasks.labeling import BiLSTM_Model, BiGRU_Model
import tensorflow as tf
from configparser import ConfigParser
import shutil
from tensorflow.python import keras
from math import ceil
from clr_callback import *
from callback import SaveMinLoss
import datetime
import kashgari
from tensorflow.python.keras.utils import get_file
from kashgari.macros import DATA_PATH
import pandas as pd
import MeCab
import logging

def set_sess_cfg():
    config_sess = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config_sess.gpu_options.allow_growth = True
    sess = tf.Session(config=config_sess)
    K.set_session(sess)

def preprocess(file):
    df = pd.read_csv(file)
    mecab = MeCab.Tagger("-Owakati")
    #result = mecab.parse(text)
    x = df['content'].map(lambda x:mecab.parse(x).split(' ')).tolist()
    y = df['target'].values.tolist()
    return x, y

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_fold = cp["TRAIN"].get("output_fold")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    sequence_length_max = cp["TRAIN"].getint("sequence_length_max")
    output_model_name = cp["TRAIN"].get("output_model_name")
    save_weights_only = cp["TRAIN"].getboolean("save_weights_only")
    cyclicLR_mode = cp["TRAIN"].get("cyclicLR_mode")
    base_lr = cp["TRAIN"].getfloat("base_lr")
    max_lr = cp["TRAIN"].getfloat("max_lr")
    file_train = cp["TRAIN"].get("file_train")
    file_valid = cp["TRAIN"].get("file_valid")
    file_test = cp["TRAIN"].get("file_test")

    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    output_dir = os.path.join('experiments', formatted_today, output_fold)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir_src = os.path.join(output_dir, 'src')
    if not os.path.isdir(output_dir_src):
        os.makedirs(output_dir_src)
    print(f"backup config file to {output_dir_src}")
    shutil.copy(config_file, os.path.join(output_dir_src, os.path.split(config_file)[1]))
    train_file = os.path.basename(__file__)
    shutil.copy(train_file, os.path.join(output_dir_src, train_file))
    logging.basicConfig(level='DEBUG')
    bert_path = get_file('bert_sample_model',
                         "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                         cache_dir=DATA_PATH,
                         untar=True)

    train_x, train_y = preprocess(file_train)
    validate_x, validate_y = preprocess(file_valid)
    test_x, test_y = preprocess(file_test)

    #'bert-large-cased'
    embedding = BERTEmbedding(bert_path,
                              sequence_length=sequence_length_max,
                              task=kashgari.CLASSIFICATION,
                              trainable=True,
                              layer_nums=4)
    #embedding = BERTEmbedding('/home/ys1/pretrained_models/BERT/Japanese_L-12_H-768_A-12_E-30_BPE/', sequence_length=sequence_length_max, task=kashgari.CLASSIFICATION)

    # 还可以选择 CNNModel CNNLSTMModel
    # model = BiGRU_Model(embedding)
    hyper = BiLSTM_Model.get_default_hyper_parameters()
    print(f'hyper parameters is:{hyper}')
    #hyper parameters is:{'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_dense': {'activation': 'softmax'}}
    # hyper['layer_bi_lstm']['units'] = 32
    model = BiLSTM_Model(embedding, hyper_parameters=hyper)
    # model.build_model(train_x, train_y)
    # model.build_multi_gpu_model(gpus=2)
    # print(model.summary())

    if save_weights_only:
        model_weights = os.path.join(output_dir, output_weights_name)
    else:
        model_weights = os.path.join(output_dir, output_model_name)

    checkpoint = keras.callbacks.ModelCheckpoint(
        model_weights,
        save_weights_only=save_weights_only,
        save_best_only=True,
        verbose=1,
    )
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
    csv_logger = keras.callbacks.CSVLogger(os.path.join(output_dir, 'training.csv'))
    batch_size_cycliclr = ceil(len(train_x)/batch_size)
    if cyclicLR_mode == 'exp_range':
        gamma = 0.99994
    else:
        gamma = 1.
    clr = CyclicLR(mode=cyclicLR_mode, step_size=batch_size_cycliclr, base_lr=base_lr, max_lr=max_lr, gamma=gamma)
    save_min_loss = SaveMinLoss(filepath=output_dir)
    tb = keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size, update_freq=1000)
    # 这是 Kashgari 内置回调函数，会在训练过程计算精确度，召回率和 F1
    eval_callback = EvalCallBack(kash_model=model,
                                 valid_x=validate_x,
                                 valid_y=validate_y,
                                 step=5)
    callbacks = [
        eval_callback,
        checkpoint,
        tb,
        csv_logger,
        # clr,
        save_min_loss,
        earlystop,
    ]

    print("** start training **")
    model.fit(train_x,
              train_y,
              x_validate=validate_x,
              y_validate=validate_y,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              fit_kwargs={
                          'workers': generator_workers,
                          'use_multiprocessing': True,
                          'class_weight': 'auto',
                          }
              )

    model_path = os.path.join(output_dir, 'model')
    model.save(model_path)
    report_evaluate = model.evaluate(test_x, test_y, debug_info=True)

    with open(os.path.join(output_dir, 'report_evaluate.log'), 'w') as f:
        f.write(f"The evaluate report is : \n{str(report_evaluate)}")

if __name__ == "__main__":
    set_sess_cfg()
    main()