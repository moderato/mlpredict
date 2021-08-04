import json
import joblib
import pkg_resources
import os

from mlpredict.prediction import predict_walltime
from mlpredict.import_tools import import_gpu


class dnn(dict):
    """Class for deep neural network architecture"""

    def __init__(self, input_dimension, input_size):
        self['layers'] = {}
        self['input'] = {}
        self['input']['dimension'] = input_dimension
        self['input']['size'] = input_size if (isinstance(input_size, tuple) and len(input_size) == 2) else (input_size, input_size)

    def save(self, path):
        """Save dnn to path"""
        if not os.path.isdir(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, 'w') as json_file:
            json.dump(self, json_file, indent=4)

    def describe(self):
        """Prints a description of of the class instance"""
        print("===========================================")
        print('%d layer network\n' % (len(self['layers'])))
        print('Input size %dx%dx%d\n'
              % (self['input']['size'][0], self['input']['size'][1],
                 self['input']['dimension']))
        for layer in self['layers']:
            print('%s (%s), now %dx%d with %d channels'
                  % (self['layers'][layer]['name'],
                     self['layers'][layer]['type'],
                     self['layers'][layer]['output_size'][0],
                     self['layers'][layer]['output_size'][1],
                     self['layers'][layer]['channels_out']))

    def add_layer(self, layer_type, layer_name, **kwargs):
        """Adds a layer to the class instance
        Args:
            layer_type: Type of layer ('Convolution', 'Fully_connected' or
                    'Max_pool')
            layer_name: Name of layer (string)
        Layer type specific args:
            Convolution:
                kernelsize
                channels_out
                padding
                strides
                use_bias
                activation
            Max_pool:
                pool_size
                strides
                padding
            Fully_connected:
        """

        num_layers = len(self['layers'])
        new_layer = num_layers + 1
        if num_layers == 0:
            input_dimension = self['input']['dimension']
            input_size = self['input']['size']
        elif 'from_layer' in kwargs.keys():
            from_layer = kwargs['from_layer']
            if isinstance(from_layer, list):
                assert all(f <= num_layers for f in from_layer)
                input_dimension = sum(self['layers'][f]['channels_out'] for f in from_layer)
                input_size = self['layers'][from_layer[0]]['output_size']
            else:
                assert from_layer <= num_layers
                input_dimension = self['layers'][from_layer]['channels_out']
                input_size = self['layers'][from_layer]['output_size']
        else:
            input_dimension = self['layers'][num_layers]['channels_out']
            input_size = self['layers'][num_layers]['output_size']

        self['layers'][new_layer] = {}     # Create new layer
        self['layers'][new_layer]['name'] = layer_name
        self['layers'][new_layer]['type'] = layer_type

        kwargs['strides'] = kwargs['strides'] if (isinstance(kwargs['strides'], tuple) and len(kwargs['kernelsize']) == 2) else (kwargs['strides'], kwargs['strides'])
        if layer_type.lower() == 'convolution':
            kwargs['kernelsize'] = kwargs['kernelsize'] if (isinstance(kwargs['kernelsize'], tuple) and len(kwargs['kernelsize']) == 2) else (kwargs['kernelsize'], kwargs['kernelsize'])
            padding_reduction = (
                (kwargs['padding'].lower() == 'valid') * (kwargs['kernelsize'][0] - 1), 
                (kwargs['padding'].lower() == 'valid') * (kwargs['kernelsize'][1] - 1)
            )
            output_size = (
                (input_size[0] - padding_reduction[0]) / kwargs['strides'][0],
                (input_size[1] - padding_reduction[1]) / kwargs['strides'][1],
            )

            self['layers'][new_layer]['matsize'] = input_size
            self['layers'][new_layer]['kernelsize'] = kwargs['kernelsize']
            self['layers'][new_layer]['channels_in'] = input_dimension
            self['layers'][new_layer]['channels_out'] = kwargs['channels_out']
            self['layers'][new_layer]['padding'] = kwargs['padding']
            self['layers'][new_layer]['strides'] = kwargs['strides']
            self['layers'][new_layer]['use_bias'] = kwargs['use_bias']
            self['layers'][new_layer]['activation'] = kwargs['activation']
            self['layers'][new_layer]['output_size'] = output_size

        if layer_type.lower() == 'max_pool':
            kwargs['pool_size'] = kwargs['pool_size'] if (isinstance(kwargs['pool_size'], tuple) and len(kwargs['pool_size']) == 2) else (kwargs['pool_size'], kwargs['pool_size'])
            padding_reduction = (
                (kwargs['padding'].lower() == 'valid') * (kwargs['pool_size'][0] - 1), 
                (kwargs['padding'].lower() == 'valid') * (kwargs['pool_size'][1] - 1)
            )
            output_size = (
                (input_size[0] - padding_reduction[0]) / kwargs['strides'][0],
                (input_size[1] - padding_reduction[1]) / kwargs['strides'][1],
            )

            self['layers'][new_layer]['pool_size'] = kwargs['pool_size']
            self['layers'][new_layer]['strides'] = kwargs['strides']
            self['layers'][new_layer]['padding'] = kwargs['padding']
            self['layers'][new_layer]['output_size'] = output_size
            self['layers'][new_layer]['channels_out'] = input_dimension

        print('%s (%s), now %dx%d with %d channels'
              % (layer_name, layer_type, output_size[0], output_size[1],
                 self['layers'][new_layer]['channels_out']))

    def remove_last_layer(self):
        """Removes last layer of class instance"""
        num_layers = len(self['layers'])
        if num_layers > 0:
            del self['layers'][num_layers]

    def predict(self,
                gpu,
                optimizer='SGD',
                batchsize=1,
                model_file='',
                scaler_file=''):
        """Predicts execution time of class instance
        Args:
            gpu: can be local json file with GPU definition or
            batchsize: default 1
            saved_model: tensorflow model, by default uses model from
                    all GPUs
            scaler_file: sklearn scaler used to normalise inputs to
                    tensorflow model
        Returns:
            total execution time
            layer names
            layer execution times
        """

        if model_file == '':
            model_file = pkg_resources.resource_filename(
                'mlpredict', 'model/model_all')
        if scaler_file == '':
            scaler_file = pkg_resources.resource_filename(
                'mlpredict', 'model/scaler_Conv_all.save')

        gpu_stats = import_gpu(gpu)

        scaler = joblib.load(scaler_file)

        layer, time = predict_walltime(
            self, model_file, scaler, batchsize, optimizer,
            gpu_stats['bandwidth'], gpu_stats['cores'], gpu_stats['clock'])

        return sum(time), layer, time

    def get_num_layers(self):
        return len(self['layers'])
