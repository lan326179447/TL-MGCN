from wdl import InitParam2
from wdl import train_experiment2_tr

#实例化参数类
param2 = InitParam2(filename="Datasets\\A1.csv", n_iters=310, batch_size=22)
#设置记录实验结果的文件
param2.set_save_record_file(filename="tr_record.csv")
#设置保存模型文件
#param2.set_save_model_file(weights_file="tr_A1_trained_weights.pkl", model_file="tr_A1_RF.pkl")
#训练模型
train_experiment2_tr(source_weights="Models\\Models_SD\\trained_weights4AS1.pkl", param=param2)
