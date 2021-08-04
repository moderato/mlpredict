import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

import mlpredict


gpu = 'V100'
opt = 'SGD'


model_names = ["ResNet50", "Inception-V3"]
models = []
for m in model_names:
    models.append(mlpredict.dnn.create_model(m, save=True))
    
    print("=====================")


for model, name in zip(models, model_names):
    print("======== {} ========".format(name))
    
    batchsize = [2 ** b for b in range(0, 7)]
    time_layer = []
    time_total = []

    for i in range(len(batchsize)):
        t_total, layer, t_layer = model.predict(gpu = gpu,
                                              optimizer = opt,
                                              batchsize = batchsize[i])
        time_layer.append(t_layer)
        time_total.append(t_total)
        
        print("Batch size: {}, total time: {:.2f}".format(batchsize[i], t_total[0]))
