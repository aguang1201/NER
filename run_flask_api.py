# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: run_flask_api
@time: 2019-02-24

"""
import random
from flask import Flask, jsonify
from kashgari.tasks.seq_labeling import BLSTMCRFModel
from configparser import ConfigParser
import os
from bert_blstm_crf_eng_train import set_sess_cfg


app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_tasks():
    y = model.predict(sentence_list, output_dict=True)
    return jsonify({'x': sentence_list, 'y': y})


if __name__ == '__main__':
    set_sess_cfg()
    # must run predict once before `app.run` to prevent predict error
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
    model.predict(sentence_list)
    app.run(debug=True, port=8080)
