import tensorflow as tf
# To avoid Blas xGEMM error with TF2
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np


def predict_walltime(model,
                     model_file,
                     scaler,
                     batchsize,
                     optimizer,
                     bandwidth,
                     cores,
                     clock):
    """Predicts execution time of deep neuronal network on some hardware
    Args:
        model: Deep neural network architecture, instance of the model class
        model_file: tensorflow model
        sklearn: skleran scaler
        batchsize (int)
        optimizer (string)
        bandwidth: GPU memory bandwidth in GB/s (int)
        cores: Number of GPU cores (int)
        clock: GPU clock frequency in MHz (int)
    Returns:
        layer_name
        layer_prediction: Predicted execution time of layer_name
    """

    tf.compat.v1.enable_resource_variables()
    pb_model = tf.saved_model.load(model_file)
    tf.compat.v1.enable_resource_variables()
    f = pb_model.signatures["serving_default"]

    layer_prediction = []
    layer_name = []

    for layer in model['layers']:
        if model['layers'][layer]['type'] == 'Convolution':
            features = get_input_features(model['layers'][layer],
                                              scaler,
                                              batchsize,
                                              optimizer,
                                              bandwidth,
                                              cores,
                                              clock)

            result = f(model_input=tf.constant(features, dtype=tf.float32), model_istraining=tf.constant(False))
            layer_prediction.append(result['model_prediction'])
            layer_name.append(model['layers'][layer]['name'])

    return layer_name, layer_prediction


def get_input_features(
        dictionary,
        scaler,
        batchsize,
        optimizer,
        bandwidth,
        cores,
        clock):

    features = np.array([batchsize,
                         np.prod(dictionary['matsize']),
                         np.prod(dictionary['kernelsize']),
                         dictionary['channels_in'],
                         dictionary['channels_out'],
                         (1 if dictionary['padding'].lower() == 'same' else 0),
                         dictionary['strides'][0],
                         dictionary['use_bias'],
                         (1 if optimizer.lower() == 'sgd' else 0),
                         (1 if optimizer.lower() == 'adadelta' else 0),
                         (1 if optimizer.lower() == 'adagrad' else 0),
                         (1 if optimizer.lower() == 'momentum' else 0),
                         (1 if optimizer.lower() == 'adam' else 0),
                         (1 if optimizer.lower() == 'rmsprop' else 0),
                         (1 if dictionary['activation'].lower() == 'relu' else 0),
                         (1 if dictionary['activation'].lower() == 'tanh' else 0),
                         (1 if dictionary['activation'].lower() == 'sigmoid' else 0),
                         bandwidth,
                         cores,
                         clock])

    features = scaler.transform(features.reshape(1, -1))
    return features
