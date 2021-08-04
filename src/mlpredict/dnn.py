import mlpredict


def create_vgg16():
    VGG16 = mlpredict.api.dnn(input_dimension=3, input_size=224)

    VGG16.add_layer('Convolution', 'conv1_1', kernelsize=3, channels_out=64, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv1_2', kernelsize=3, channels_out=64, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Max_pool', 'pool1', pool_size=2, padding='SAME', strides=2)


    VGG16.add_layer('Convolution', 'conv2_1', kernelsize=3, channels_out=128, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv2_2', kernelsize=3, channels_out=128, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Max_pool', 'pool2', pool_size=2, padding='SAME', strides=2)


    VGG16.add_layer('Convolution', 'conv3_1', kernelsize=3, channels_out=256, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv3_2', kernelsize=3, channels_out=256, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv3_3', kernelsize=3, channels_out=256, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Max_pool', 'pool3', pool_size=2, padding='SAME', strides=2)


    VGG16.add_layer('Convolution', 'conv4_1', kernelsize=3, channels_out=512, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv4_2', kernelsize=3, channels_out=512, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv4_3', kernelsize=3, channels_out=512, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Max_pool', 'pool4', pool_size=2, padding='SAME', strides=2)


    VGG16.add_layer('Convolution', 'conv5_1', kernelsize=3, channels_out=512, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv5_2', kernelsize=3, channels_out=512, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'conv5_3', kernelsize=3, channels_out=512, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Max_pool', 'pool5', pool_size=2, padding='SAME', strides=2)


    VGG16.add_layer('Convolution', 'fc6', kernelsize=7, channels_out=4096, 
                    padding='VALID', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'fc7', kernelsize=1, channels_out=4096, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')
    VGG16.add_layer('Convolution', 'fc8', kernelsize=1, channels_out=1000, 
                    padding='SAME', strides=1, use_bias=1, activation='relu')

    return VGG16


def create_resnet50():
    resnet50 = mlpredict.api.dnn(input_dimension=3, input_size=224)
    resnet50.add_layer('Convolution', 'conv1', kernelsize=7, channels_out=64, 
                    padding='SAME', strides=2, use_bias=0, activation='relu')
    resnet50.add_layer('Max_pool', 'max_pool', pool_size=3, padding='SAME', strides=2)

    block_rep = [3, 4, 6, 3]
    for idx, r in enumerate(block_rep):
        for b in range(0, r):
            downsample = (idx > 0 and b == 0)
            from_layer = resnet50.get_num_layers()
            resnet50.add_layer('Convolution', 'conv{}_{}_1'.format(idx+2, b), kernelsize=1, channels_out=64*(2**idx), 
                            padding='SAME', strides=1, use_bias=0, activation='relu')
            resnet50.add_layer('Convolution', 'conv{}_{}_2'.format(idx+2, b), kernelsize=3, channels_out=64*(2**idx), 
                            padding='SAME', strides=2 if downsample else 1, use_bias=0, activation='relu')
            resnet50.add_layer('Convolution', 'conv{}_{}_3'.format(idx+2, b), kernelsize=1, channels_out=256*(2**idx), 
                            padding='SAME', strides=1, use_bias=0, activation='relu')
            # Skip add here: minor
            if downsample:
                resnet50.add_layer('Convolution', 'conv{}_{}_ds'.format(idx+2, b), kernelsize=1, channels_out=256*(2**idx), 
                                padding='SAME', strides=2, use_bias=0, activation='relu', from_layer=from_layer)
    resnet50.add_layer('Max_pool', 'avg_pool', pool_size=7, padding='VALID', strides=1) # Doesn't matter
    resnet50.add_layer('Convolution', 'fc', kernelsize=1, channels_out=1000, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')

    return resnet50


def create_iv3():
    iv3 = mlpredict.api.dnn(input_dimension=3, input_size=299)
    iv3.add_layer('Convolution', 'conv1', kernelsize=3, channels_out=32, 
                    padding='SAME', strides=2, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv2', kernelsize=3, channels_out=32, 
                    padding='VALID', strides=1, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv3', kernelsize=3, channels_out=64, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
    iv3.add_layer('Max_pool', 'max_pool', pool_size=3, padding='SAME', strides=2)
    iv3.add_layer('Convolution', 'conv4', kernelsize=3, channels_out=64, 
                    padding='VALID', strides=1, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv5', kernelsize=3, channels_out=80, 
                    padding='SAME', strides=2, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv6', kernelsize=3, channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')

    # Module A
    pool_features = [32, 64, 64]
    branch_start = iv3.get_num_layers()
    branch_ends = []
    for r in range(3):
        if len(branch_ends) > 0:
            branch_start = branch_ends
            branch_ends = [] # Skip concat
        iv3.add_layer('Convolution', 'conv7_{}_1_1'.format(r), kernelsize=1, channels_out=64, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        branch_ends.append(iv3.get_num_layers())
        iv3.add_layer('Convolution', 'conv7_{}_2_1'.format(r), kernelsize=1, channels_out=48, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        iv3.add_layer('Convolution', 'conv7_{}_2_2'.format(r), kernelsize=5, channels_out=64, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())
        iv3.add_layer('Convolution', 'conv7_{}_3_1'.format(r), kernelsize=1, channels_out=64, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        iv3.add_layer('Convolution', 'conv7_{}_3_2'.format(r), kernelsize=3, channels_out=96, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        iv3.add_layer('Convolution', 'conv7_{}_3_3'.format(r), kernelsize=3, channels_out=96, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())
        iv3.add_layer('Max_pool', 'pool7_{}_4_1'.format(r), pool_size=3, padding='SAME', strides=1, from_layer=branch_start) # No avg pool and use max pool instead
        iv3.add_layer('Convolution', 'conv7_{}_4_2'.format(r), kernelsize=1, channels_out=pool_features[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())
        
    # Grid reduction
    branch_start = branch_ends
    branch_ends = []
    iv3.add_layer('Convolution', 'conv8_1_1', kernelsize=1, channels_out=384, 
                    padding='SAME', strides=2, use_bias=0, activation='relu', from_layer=branch_start)
    branch_ends.append(iv3.get_num_layers())
    iv3.add_layer('Convolution', 'conv8_2_1', kernelsize=1, channels_out=64, 
                padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
    iv3.add_layer('Convolution', 'conv8_2_2', kernelsize=5, channels_out=96, 
                padding='SAME', strides=1, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv8_2_3', kernelsize=3, channels_out=96, 
                padding='SAME', strides=2, use_bias=0, activation='relu')
    branch_ends.append(iv3.get_num_layers())
    iv3.add_layer('Max_pool', 'pool8_3_1', pool_size=3, padding='SAME', strides=2, from_layer=branch_start) # No avg pool and use max pool instead
    branch_ends.append(iv3.get_num_layers())

    # Module B
    c7 = [128, 160, 160, 192]
    branch_start = branch_ends
    branch_ends = []
    for r in range(4):
        if len(branch_ends) > 0:
            branch_start = branch_ends
            branch_ends = [] # Skip concat
        iv3.add_layer('Convolution', 'conv9_{}_1_1'.format(r), kernelsize=1, channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        branch_ends.append(iv3.get_num_layers())
        iv3.add_layer('Convolution', 'conv9_{}_2_1'.format(r), kernelsize=1, channels_out=c7[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        iv3.add_layer('Convolution', 'conv9_{}_2_2'.format(r), kernelsize=(1,7), channels_out=c7[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        iv3.add_layer('Convolution', 'conv9_{}_2_3'.format(r), kernelsize=(7,1), channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())
        iv3.add_layer('Convolution', 'conv9_{}_3_1'.format(r), kernelsize=1, channels_out=c7[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        iv3.add_layer('Convolution', 'conv9_{}_3_2'.format(r), kernelsize=(1,7), channels_out=c7[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        iv3.add_layer('Convolution', 'conv9_{}_3_3'.format(r), kernelsize=(7,1), channels_out=c7[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        iv3.add_layer('Convolution', 'conv9_{}_3_4'.format(r), kernelsize=(1,7), channels_out=c7[r], 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        iv3.add_layer('Convolution', 'conv9_{}_3_5'.format(r), kernelsize=(7,1), channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())
        iv3.add_layer('Max_pool', 'pool9_{}_4_1'.format(r), pool_size=3, padding='SAME', strides=1, from_layer=branch_start) # No avg pool and use max pool instead
        iv3.add_layer('Convolution', 'conv9_{}_4_2'.format(r), kernelsize=1, channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())
        
    # Grid reduction
    branch_start = branch_ends
    branch_ends = []
    iv3.add_layer('Convolution', 'conv10_1_1', kernelsize=1, channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
    iv3.add_layer('Convolution', 'conv10_1_2', kernelsize=3, channels_out=320, 
                    padding='SAME', strides=2, use_bias=0, activation='relu')
    branch_ends.append(iv3.get_num_layers())
    iv3.add_layer('Convolution', 'conv10_2_1', kernelsize=1, channels_out=192, 
                padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
    iv3.add_layer('Convolution', 'conv10_2_2', kernelsize=(1,7), channels_out=192, 
                padding='SAME', strides=1, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv10_2_3', kernelsize=(7,1), channels_out=192, 
                padding='SAME', strides=1, use_bias=0, activation='relu')
    iv3.add_layer('Convolution', 'conv10_2_4', kernelsize=3, channels_out=192, 
                padding='SAME', strides=2, use_bias=0, activation='relu')
    branch_ends.append(iv3.get_num_layers())
    iv3.add_layer('Max_pool', 'pool10_4_1', pool_size=3, padding='SAME', strides=2, from_layer=branch_start) # No avg pool and use max pool instead
    branch_ends.append(iv3.get_num_layers())

    # Block C
    branch_start = branch_ends
    branch_ends = []
    for r in range(2):
        if len(branch_ends) > 0:
            branch_start = branch_ends
            branch_ends = [] # Skip concat
        iv3.add_layer('Convolution', 'conv11_{}_1_1'.format(r), kernelsize=1, channels_out=320, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        branch_ends.append(iv3.get_num_layers())

        iv3.add_layer('Convolution', 'conv11_{}_2_1'.format(r), kernelsize=1, channels_out=384, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        tmp_branch_start = iv3.get_num_layers()
        iv3.add_layer('Convolution', 'conv11_{}_2_2'.format(r), kernelsize=(1,3), channels_out=384, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=tmp_branch_start)
        iv3.add_layer('Convolution', 'conv11_{}_2_3'.format(r), kernelsize=(3,1), channels_out=384, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=tmp_branch_start)
        branch_ends.append(iv3.get_num_layers()-1)
        branch_ends.append(iv3.get_num_layers())

        iv3.add_layer('Convolution', 'conv11_{}_3_1'.format(r), kernelsize=1, channels_out=448, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=branch_start)
        iv3.add_layer('Convolution', 'conv11_{}_3_2'.format(r), kernelsize=3, channels_out=384, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        tmp_branch_start = iv3.get_num_layers()
        iv3.add_layer('Convolution', 'conv11_{}_3_3'.format(r), kernelsize=(1,3), channels_out=384, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=tmp_branch_start)
        iv3.add_layer('Convolution', 'conv11_{}_3_4'.format(r), kernelsize=(3,1), channels_out=384, 
                    padding='SAME', strides=1, use_bias=0, activation='relu', from_layer=tmp_branch_start)
        branch_ends.append(iv3.get_num_layers()-1)
        branch_ends.append(iv3.get_num_layers())

        iv3.add_layer('Max_pool', 'pool11_{}_4_1'.format(r), pool_size=3, padding='SAME', strides=1, from_layer=branch_start) # No avg pool and use max pool instead
        iv3.add_layer('Convolution', 'conv11_{}_4_2'.format(r), kernelsize=1, channels_out=192, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')
        branch_ends.append(iv3.get_num_layers())

    iv3.add_layer('Max_pool', 'avg_pool'.format(r), pool_size=8, padding='VALID', strides=1, from_layer=branch_ends) # No avg pool and use max pool instead
    iv3.add_layer('Convolution', 'fc', kernelsize=1, channels_out=1000, 
                    padding='SAME', strides=1, use_bias=0, activation='relu')

    return iv3


def create_model(name, save=False):
    assert name in ['VGG16', 'ResNet50', 'Inception-V3']
    if name == 'VGG16':
        model = create_vgg16()
    elif name == 'ResNet50':
        model = create_resnet50()
    else: # Inception-V3
        model = create_iv3()
    
    if save:
        model.save('dnn_architecture/{}.json'.format(name))
    
    return model
