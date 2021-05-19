import pandas as pd
import pickle
import numpy as np
import csv
import os
from wdl.build_wdl_fp_v2 import build_wdl_fingerprint_fun
from wdl.run_utils import read_csv, save_result_to_csv, save_result_to_csv2

def predict_fp_and_pvalue(input_file,weights_file,model_file):
    """
    输出分子指纹和预测的生物活性
    :param input_file: 待预测的配体分子的SMILES分子式csv文件
    :param weights_file: 训练好的wdl网络的权重值pkl文件
    :param model_file: 训练好的RF模型pkl文件
    :return:
    """
    #print("Loading data...")
    #print('读取待预测文件：',input_file)
    input_smiles = read_csv(input_file)
    #print(input_smiles)
    model_params = dict(fp_length=50,
                        fp_depth=4,
                        hidden_width=20,
                        h1_size=100,
                        n_estimators=100,
                        max_features='log2',
                        L2_reg=np.exp(-2))

    def build_weight_fp_experiment(input_smiles,init_weight):
        conv_layer_sizes = [model_params['hidden_width']] * model_params['fp_depth']  # [20,20,20,20].
        conv_arch_params = {'num_hidden_features': conv_layer_sizes,
                            'fp_length': model_params['fp_length'], 'normalize': 1}
        conv_fp_func, conv_parser = build_wdl_fingerprint_fun(**conv_arch_params)
        input_fp = conv_fp_func(init_weight, input_smiles)
        return input_fp

    #print("读取权重文件:",weights_file)
    with open(weights_file, 'rb') as fr:
        trained_weights = pickle.load(fr)
    fp = build_weight_fp_experiment(input_smiles,trained_weights)

    #print('读取模型文件',model_file)
    with open(model_file, 'rb') as fr:
        rf_model = pickle.load(fr)
    p_value = rf_model.predict(fp)

    return fp,p_value


def trans(target,smiles1):
    #target = target.split()[-1][1:7]
    target1 = []
    for i in smiles1:
        target1.append(target)

    if os.path.exists("usr_input.csv"):
        os.remove("usr_input.csv")
    df = pd.DataFrame(columns=["GPCR","smiles"],index=None)
    df["GPCR"] = target1
    df["smiles"] = smiles1
    df.to_csv("usr_input.csv",index=False)

    temp = df['GPCR'][0]
    #设置wdl网络权重文件
    trained_weights_file = 'Models/Models_TD/tr_'+temp+'_trained_weights.pkl'
    #设置rf模型文件
    model_file = 'Models/Models_TD/'+'tr_'+temp+'_RF.pkl'
    #调用函数做预测+
    fp, pvalue = predict_fp_and_pvalue('usr_input.csv', weights_file=trained_weights_file, model_file=model_file)
    #设置保存结果csv文件
    save_file = 'Result/'+'tr_'+temp+'_result.csv'
    d = pd.read_csv(save_file)
    print(d[{'GPCR', 'smiles', 'predict_value'}])
    #将结果保存到csv文件中
    save_result_to_csv2('usr_input.csv', outputfile=save_file, pvalue=pvalue, fp=fp)

def transfile(target,uploadfile):
    #target = target.split()[-1][1:7]
    target1 = []

    df0 = pd.read_csv(uploadfile, names=["smiles"], index_col=None)
    smiles1 = []
    smiles = df0["smiles"]
    for i in smiles:
        smiles1.append(i)
    for j in smiles1:
        target1.append(target)

    if os.path.exists("usr_input1.csv"):
        os.remove("usr_input1.csv")
    df = pd.DataFrame(columns=["GPCR","smiles"],index=None)
    df["GPCR"] = target1
    df["smiles"] = smiles1
    df.to_csv("usr_input1.csv",index=False)

    temp = df['GPCR'][0]
    #设置wdl网络权重文件
    trained_weights_file = 'Models/Models_TD/tr_'+temp+'_trained_weights.pkl'
    #设置rf模型文件
    model_file = 'Models/Models_TD/'+'tr_'+temp+'_RF.pkl'
    #调用函数做预测+
    fp, pvalue = predict_fp_and_pvalue('usr_input1.csv', weights_file=trained_weights_file, model_file=model_file)
    #设置保存结果csv文件
    save_file = 'Result/'+'tr_'+temp+'_result.csv'
    d=pd.read_csv(save_file)
    print(d[{'GPCR','smiles','predict_value'}])
    #将结果保存到csv文件中
    save_result_to_csv2('usr_input1.csv', outputfile=save_file, pvalue=pvalue, fp=fp)

trans("P47900",["CC1=CC=CC(=C1)C2=NOC(=N2)CN(C(C)C)C(=O)C3=CC(=CC(=C3)OC)C","C1=CC=C2C=C(C=CC2=C1)C=CC(=O)CCC(=O)O"])

#transfile("P30939","data.csv")
