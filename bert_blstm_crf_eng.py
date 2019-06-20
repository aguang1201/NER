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
    output_fold = cp["TRAIN"].get("output_fold")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    output_model_name = cp["TRAIN"].get("output_model_name")
    save_weights_only = cp["TRAIN"].getboolean("save_weights_only")
    cyclicLR_mode = cp["TRAIN"].get("cyclicLR_mode")
    base_lr = cp["TRAIN"].getfloat("base_lr")
    max_lr = cp["TRAIN"].getfloat("max_lr")

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

    train_x, train_y = CoNLL2003Corpus.get_sequence_tagging_data('train')
    validate_x, validate_y = CoNLL2003Corpus.get_sequence_tagging_data('validate')
    test_x, test_y = CoNLL2003Corpus.get_sequence_tagging_data('test')

    #'bert-large-cased'
    embedding = BERTEmbedding('bert-large-cased', 30)
    # 还可以选择 `BLSTMModel` 和 `CNNLSTMModel`
    model = BLSTMCRFModel(embedding)
    # model.build_model(train_x, train_y)
    # model.build_multi_gpu_model(gpus=2)
    # print(model.summary())

    if save_weights_only:
        model_weights = os.path.join(output_dir, output_weights_name)
    else:
        model_weights = os.path.join(output_dir, output_model_name)

    checkpoint = ModelCheckpoint(
        model_weights,
        save_weights_only=save_weights_only,
        save_best_only=True,
        verbose=1,
    )
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
    csv_logger = CSVLogger(os.path.join(output_dir, 'training.csv'))
    batch_size_cycliclr = ceil(len(train_x)/batch_size)
    if cyclicLR_mode == 'exp_range':
        gamma = 0.99994
    else:
        gamma = 1.
    clr = CyclicLR(mode=cyclicLR_mode, step_size=batch_size_cycliclr, base_lr=base_lr, max_lr=max_lr, gamma=gamma)
    save_min_loss = SaveMinLoss(filepath=output_dir)
    tb = TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size)
    callbacks = [
        checkpoint,
        tb,
        csv_logger,
        clr,
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
              labels_weight=True,
              fit_kwargs={'callbacks': callbacks,
                          'workers': generator_workers,
                          'use_multiprocessing': True,
                          'class_weight': 'auto',
                          })

    model_path = os.path.join(output_dir, 'model')
    model.save(model_path)
    report_evaluate = model.evaluate(test_x, test_y, debug_info=True)

    with open(os.path.join(output_dir, 'report_evaluate.log'), 'w') as f:
        f.write(f"The evaluate report is : \n{str(report_evaluate)}")

if __name__ == "__main__":
    set_sess_cfg()
    main()