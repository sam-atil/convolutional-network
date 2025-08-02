'''network.py
Represents  a neural network (collection of layers)
Samuel
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import time

import accelerated_layer
import filter_ops
import layer
import numpy as np


class Network:
    '''Represents a neural network with some number of layers of various types.
    To create a specific network, create a subclass (e.g. ConvNet4) then
    add layers to it. For this project, the focus will be on building the
    ConvNet4 network.
    '''
    def __init__(self, reg=0, verbose=True):
        '''Method is pre-filled for you (shouldn't require modification).'''
        # Python list of Layer object references that make up out network
        self.layers = []
        # Regularization strength
        self.reg = reg
        # Whether we want network-related debug/info print outs
        self.verbose = verbose

        # Python list of ints. These are the indices of layers in `self.layers`
        # that have network weights.
        self.wt_layer_inds = []

        # Indices of all dropout layers in the layers list
        self.dropout_layer_inds = []

        # Are we currently training?
        self.in_train_mode = False

        # As in former projects, Python list of loss, training/validation
        # accuracy during training recorded at some frequency (e.g. every epoch)
        self.loss_history = []
        self.train_acc_history = []
        self.validation_acc_history = []


    def is_training(self):
        '''Is the CNN currently training?

        Returns:
        --------
        bool. Whether net is training.
        '''
        
        return self.in_train_mode

    def compile(self, optimizer_name, **kwargs):
        '''Tells each network layer how weights should be updated during backprop
        during training (e.g. stochastic gradient descent, adam, etc.)

        This method is pre-filled for you (shouldn't require modification).

        NOTE: This NEEDS to be called AFTER creating your ConvNet4 object,
        but BEFORE you call `fit()` to train the net (otherwise, how does your
        net know how to update the weights?).

        Parameters:
        ----------
        optimizer_name: string. Name of optimizer class to use to update wts.
            See optimizer::create_optimizer for specific ones supported.
        **kwargs: Any number of optional parameters that get passed to the
            optimizer of your choice. e.g. learning rate.
        '''
        # Only set an optimizer for each layer with weights
        for l in [self.layers[i] for i in self.wt_layer_inds]:
            l.compile(optimizer_name, **kwargs)

    def set_dropout_layers_mode(self, in_train_mode=False):
        '''Relays whether we are in training mode to dropout layers only.

        Parameters:
        -----------
        in_train_mode: bool.
            Are we currently training the net?

        TODO:
        1. Update the network's mode.
        2. Update each of the dropout layer's mode.
        '''
        self.in_train_mode = in_train_mode

        for index in self.dropout_layer_inds:
            self.layers[index].set_mode(in_train_mode)


    def forward(self, inputs, y):
        '''Do forward pass through whole network

        Parameters:
        ----------
        inputs: ndarray.
            Inputs coming into the input layer of the net. shape=(B, n_chans, img_y, img_x)
        y: ndarray.
            int-coded class assignments of training mini-batch. 0,...,numClasses-1 shape=(B,) for mini-batch size B.

        Returns:
        -------
        loss: float.
            REGULARIZED loss.

        TODO:
        1. Call the forward method of each layer in the network.
            Make the output of the previous layer the input to the next.
        2. Compute and get the loss from the LAST network layer.
        2. Compute and get the weight regularization via `self.wt_reg_reduce()` (implement this next)
        4. Return the sum of the loss and the regularization term.
        '''

        #Forward function in layer class computes both netIn and netAct, returns netAct
        data = inputs
        for layer in self.layers: #loops through each layer in the network
            data = layer.forward(data) #calls on the forward function of the current layer
        loss = self.layers[-1].loss(y) #calculates the loss from the output/dense layer
        wt_reg = self.wt_reg_reduce()

        #print("Finished the forward Pass\n")

        return loss + wt_reg



    def wt_reg_reduce(self):
        '''Computes the loss weight regularization for all network layers that have weights

        Returns:
        -------
        wt_reg: float. Regularization for weights from all layers across the network.

        NOTE: You only can compute regularization for layers with wts!
        Layer indicies with weights are maintained in `self.wt_layer_inds`.
        The network regularization `wt_reg` is simply the sum of all the regularization terms
        for each individual layer.
        '''
        
        wt_reg = 0.0 #empty variable
        for index in self.wt_layer_inds:
            wt_reg += (self.reg/2) * np.sum(self.layers[index].get_wts() ** 2)

        return wt_reg
        



    def backward(self, y):
        '''Initiates the backward pass through all the layers of the network.

        Parameters:
        ----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -------
        None

        TODO:
        1. Initialize d_upstream, d_wts, d_b to None.
        2. Loop through the network layers in REVERSE ORDER, calling the `Layer` backward method.
            Remember that the output of layer.backward() becomes the d_upstream to the next layer down.
            We don't care about d_wts, d_b in this method (computed/stored in Layer).
        '''
        
        d_upstream = None
        d_wts = None
        d_b = None

        for layer in reversed(self.layers): #Reverse For Loop for layers
            d_upstream, d_wts, d_b = layer.backward(d_upstream, y)

    def predict(self, inputs):
        '''Classifies novel inputs presented to the network using the current
        weights.

        Parameters:
        ----------
        inputs: ndarray. shape=shape=(num test samples, n_chans, img_y, img_x)
            This is test data.

        Returns:
        -------
        pred_classes: ndarray. shape=shape=(num test samples)
            Predicted classes (int coded) derived from the network.

        TODO:
        1. Do the forward pass thru the net.
        2. Before doing the forward pass, configure the dropout layers to operate in predict/test mode.
        3. Afer doing the forward pass, configure the dropout layers to operate in whatever mode they were at the start
        of this method.
        '''
        
        #Configuring the Dropout layers to False
        cur_mode = self.is_training()
        self.set_dropout_layers_mode(False)

        #Conducting a forward pass on the inputs(Allows us to avoid loss issues)
        data = inputs
        for layer in self.layers:
            data = layer.forward(data)

        #creating the predictive classes based off the last layer's net_in
        last_layer = self.layers[-1].net_in
        pred_classes = last_layer.argmax(axis = 1)

        #Configuring Dropout layers back to original mode
        self.set_dropout_layers_mode(cur_mode)

        return pred_classes

    def accuracy(self, inputs, y, samp_sz=500, mini_batch_sz=15):
        '''Computes accuracy using current net on the inputs `inputs` with classes `y`.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        ----------
        inputs: ndarray. shape=shape=(num samples, n_chans, img_y, img_x)
            We are testing the classification accuracy on these data.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(N,) for mini-batch size N.
        samp_sz: int. If the number of samples is bigger than this number,
            we take a random sample from `inputs` of this size. We do this to
            keep performance of this method reasonable.
        mini_batch_sz: Because it might be tricky to hold all the training
            instances in memory at once, process and evaluate the accuracy of
            samples from `input` in mini-batches. We merge the accuracy scores
            across batches so the result is no different than processing all at
            once.
        '''
        n_samps = len(inputs)

        # Do we subsample the input?
        if n_samps > samp_sz:
            rng = np.random.default_rng(0)
            subsamp_inds = rng.choice(n_samps, samp_sz)
            n_samps = samp_sz
            inputs = inputs[subsamp_inds]
            y = y[subsamp_inds]

        # How many mini-batches do we split the data into to test accuracy?
        n_batches = int(np.ceil(n_samps / mini_batch_sz))
        # Placeholder for our predicted class ints
        y_pred = np.zeros(len(inputs), dtype=np.int32)

        # Compute the accuracy through the `predict` method in batches.
        # Strategy is to use a 1D cursor `b` to extract a range of inputs to process (a mini-batch)
        for b in range(n_batches):
            low = b*mini_batch_sz
            high = b*mini_batch_sz+mini_batch_sz
            # Tolerate out-of-bounds as we reach the end of the num samples
            if high > n_samps:
                high = n_samps

            # Get the network predicted classes and put them in the right place
            y_pred[low:high] = self.predict(inputs[low:high])

        # Accuracy is the average proportion that the prediction matchs the true int class
        acc = np.mean(y_pred == y)

        return acc

    def fit(self, x_train, y_train, x_validate, y_validate, mini_batch_sz=256, n_epochs=1, acc_freq=10, print_every=50):
        '''Trains the neural network on data

        Parameters:
        ----------
        x_train: ndarray. shape=(num training samples, n_chans, img_y, img_x).
            Training data.
        y_train: ndarray. shape=(num training samples,).
            Training data classes, int coded.
        x_validate: ndarray. shape=(num validation samples, n_chans, img_y, img_x).
            Every so often during training (see acc_freq param), we compute
            the accuracy of the network in classifying the validation set
            (out-of-training-set generalization). This is the data we use.
        y_validate: ndarray. shape=(num validation samples,).
            Validation data classes, int coded.
        mini_batch_sz: int.
            Mini-batch training size.
        n_epochs: int.
            Number of training epochs.
        print_every: int.
            Controls the frequency (in iterations) with which to wait before printing out the loss
            and iteration number.
            NOTE: Previously, you used number of epochs rather than iterations to measure the frequency
            of print-outs. Use the simpler-to-implement units of iterations here because CNNs are
            more computationally intensive and you may want print-outs during an epoch.
        acc_freq: int. Should be equal to or a multiple of `print_every`.
            How many training iterations (weight updates) we wait before computing accuracy on the
            full training and validation sets?
            NOTE: This is is a computationally intensive process for the big network so make sure
            that you only COMPUTE training and validation accuracies this often
            (i.e DON'T compute them every iteration).

        Returns:
        --------
        self.loss_history: list.
            Record of training loss computed over each mini-batch of training for every epoch.
        self.train_acc_history. list.
            Record of accuracy on training set computed on entire training set computed every `acc_freq` training iters.
        self.validation_acc_history
            Record of accuracy on val set computed on val training set computed every `acc_freq` training iters.

        TODO: Complete this method's implementation.
        - It should follow the familar format for a training loop: Randomly sampling to get a mini-batch,
        doing the forward and backward pass, computing the loss, updating the wts/biases.
        - Add support for `print_every` and `acc_freq`.
        - Use the Python time module to print out the runtime (in minutes) for iteration 0 only.
        Also printout the projected time for completing ALL training iterations. (For simplicity, you don't need to
        consider the time taken for computing train and validation accuracy).
        - Remember to configure Dropout layer(s) appropriately: they should be in training mode while training and not
        in training mode when training is over.
        '''
        
        #Putting the network into Training Mode
        self.set_dropout_layers_mode(True)

        #Local Vars
        train_samps = len(x_train)
        val_samps = len(x_validate)

        num_iters = max(int(train_samps / mini_batch_sz), 1)
        
        print(f'n_epoch: {n_epochs}, iter_per_epoch: {num_iters}\n')
        total_iters = n_epochs * num_iters

        #history lists
        loss_his = []
        trnAcc_his = []
        valAcc_his = []

        iter_count = 0
        start = time.time() #To record the time
        rng = np.random.default_rng(0) #not sure if this correct

        #Running the main loop in calculating the loss values, and accuracy values
        for iteration in range(total_iters):
            #Creating the minibatch
            indices = rng.choice(train_samps, mini_batch_sz)
            batch_x = x_train[indices]
            batch_y = y_train[indices]

            #Conducting the forward and backward pass
            self.forward(batch_x, batch_y)
            self.backward(batch_y)

            #Grabbing the Loss Value
            last_layer = self.layers[-1]

            loss = last_layer.loss(batch_y)
            loss_his.append(loss)

            #updating the layer weights
            for layer in self.layers:
                layer.update_weights()
            

            #Iteration 0
            if iter_count == 0:
                one_iter = (time.time() - start) / 60 #records the elapsed time to complete one iteration
                total_time = one_iter * total_iters
                print(f'The Time at start of Fit: {start}, and the time to complete Fit: {total_time}')


            #Print_Every and Acc_Freq support
            if iter_count % acc_freq == 0:
                #print(f'Conv Layer Weights{self.layers[0].wts}')
                #calculating Train_Acc and Val_Acc
                train_acc = self.accuracy(x_train, y_train, train_samps, mini_batch_sz)
                val_acc = self.accuracy(x_validate, y_validate, val_samps, mini_batch_sz)
                trnAcc_his.append(train_acc)
                valAcc_his.append(val_acc)
                print(f'Training Accuracy{train_acc}, Validation Accuracy{val_acc}')


            if iter_count % print_every == 0:
                print(f'Iteration: {iter_count}/{total_iters}. Training Loss:{loss}')


            #Incrementing the Iter count
            iter_count += 1

        #recording global variables
        self.loss_history = loss_his
        self.train_acc_history = trnAcc_his
        self.validation_acc_history = valAcc_his

        #Taking network out of Training mode
        self.set_dropout_layers_mode(False)

        return self.loss_history, self.train_acc_history, self.validation_acc_history


class ConvNet4(Network):
    '''A minimal convolutional neural network.
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPool2D -> Flatten -> Dense -> Dense

    0. Convolution (net-in), Relu (net-act).
    1. Max pool 2D (net-in), linear (net-act).
    2. Flatten (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-2, reg=0, r_seed=None,
                 verbose=True):
        '''ConvNet4 constructor. The job of this method is to build the network as a collection of connected layers
        (in order) in `self.layers`.

        Parameters:
        -----------
        input_shape: tuple. By default: (n_chans, img_y, img_x)
            Shape of a SINGLE input sample (no mini-batch).
        n_kers: tuple.
            Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple rather than just an
            int. The reasoning is that if you wanted to create another Conv2D layer, say with 16 units, n_kers would
            then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple.
            x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST Conv2D layer.
            If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Same format as above.
            Number of hidden units in each dense layer.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple.  Same format as above.
            Pooling extent in the i-th MaxPool2D layer.
        pooling_strides: tuple.  Same format as above.
            Pooling stride in the i-th MaxPool2D layer.
        n_classes: int.
            Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float.
            Global weight scaling to use for all layers with weights
        reg: float.
            Regularization strength
        r_seed: int or None.
            Random seed for setting weights and bias parameters.
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        NOTE:
        - Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        - Number your layers starting at 0.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        #creating the layers
        conv_layer = layer.Conv2D(0, "conv2d", n_kers[0], ker_sz[0], n_chans, wt_scale, 'relu' ,reg, r_seed, verbose)
        max_layer = layer.MaxPool2D(1, "maxpool2d", pooling_sizes[0], pooling_strides[0], 'linear', reg, verbose)
        flat_layer = layer.Flatten(2, 'flatten', verbose)

        max_h = ((h - pooling_sizes[0])/pooling_strides[0]) + 1
        max_w = ((w - pooling_sizes[0])/pooling_strides[0]) + 1
        prev_units = int(n_kers[0] * max_h * max_w) #number of units in previous 2D layer collasped, like in flatten layer
        hidden_layer = layer.Dense(3, 'hidden', dense_interior_units[0], prev_units, wt_scale, 'relu', reg, r_seed, verbose)
        output_layer = layer.Dense(4, 'output', n_classes, dense_interior_units[0], wt_scale, 'softmax', reg, r_seed, verbose)

        #creating the neural network

        self.layers = [conv_layer, max_layer, flat_layer, hidden_layer, output_layer]
        self.wt_layer_inds = [0,3,4]

        print("Finished creating network")


class ConvNet4Accel(Network):
    '''A ConvNet4 network with the following layers: Conv2D -> MaxPool2D -> Flatten -> Dense -> Dense.
    This has the same architecture as ConvNet4, but uses ACCELERATED versions of your network layers (where available).

    0. Convolution (net-in), Relu (net-act).
    1. Max pool 2D (net-in), linear (net-act).
    2. Flatten (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(16,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-2, reg=0.0, r_seed=None,
                 verbose=True):
        '''ConvNet4Accel constructor. The job of this method is to build the network as a collection of connected
        layers (in order) in `self.layers`.

        Parameters:
        -----------
        input_shape: tuple. By default: (n_chans, img_y, img_x)
            Shape of a SINGLE input sample (no mini-batch).
        n_kers: tuple.
            Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple rather than just an
            int. The reasoning is that if you wanted to create another Conv2D layer, say with 16 units, n_kers would
            then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple.
            x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST Conv2D layer.
            If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Same format as above.
            Number of hidden units in each dense layer.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple.  Same format as above.
            Pooling extent in the i-th MaxPool2D layer.
        pooling_strides: tuple.  Same format as above.
            Pooling stride in the i-th MaxPool2D layer.
        n_classes: int.
            Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float.
            Global weight scaling to use for all layers with weights
        reg: float.
            Regularization strength
        r_seed: int or None.
            Random seed for setting weights and bias parameters.
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        NOTE: This should be the same as your ConvNet4, except you should use accelerated layers (where available).
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        #creating the layers(Replacing Conv2d and maxpool2d)
        conv_layer = accelerated_layer.Conv2DAccel(0, "conv2daccel", n_kers[0], ker_sz[0], n_chans, wt_scale, 'relu', reg, r_seed, verbose)
        max_layer = accelerated_layer.MaxPool2DAccel(1, "maxpool2d", pooling_sizes[0], pooling_strides[0], 'linear', reg, verbose)
        flat_layer = layer.Flatten(2, 'flatten', verbose)

        max_h = ((h - pooling_sizes[0])/pooling_strides[0]) + 1
        max_w = ((w - pooling_sizes[0])/pooling_strides[0]) + 1
        prev_units = int(n_kers[0] * max_h * max_w) #number of units in previous 2D layer collasped, like in flatten layer
        hidden_layer = layer.Dense(3, 'hidden', dense_interior_units[0], prev_units, wt_scale, 'relu', reg, r_seed, verbose)
        output_layer = layer.Dense(4, 'output', n_classes, dense_interior_units[0], wt_scale, 'softmax', reg, r_seed, verbose)

        #creating the neural network

        self.layers = [conv_layer, max_layer, flat_layer, hidden_layer, output_layer]
        self.wt_layer_inds = [0,3,4]

        print("Finished creating network")

class ConvNet4AccelV2(Network):
    '''A ConvNet4 network with the following layers: Conv2D -> MaxPool2D -> Flatten -> Dense -> Dropout -> Dense.
    This has the same architecture as ConvNet4/ConvNet4Accel, but adds a Dropout layer before the output layer.

    0. Convolution (net-in), Relu (net-act).
    1. Max pool 2D (net-in), linear (net-act).
    2. Flatten (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dropout (net-in), linear (net-act).
    5. Dense (net-in), soft-max (net-act).

    NOTE: Where available, the ACCELERATED versions of your network layers should be used in this network.
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(16,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-2, reg=0.0, dropout_rate=0.0,
                 r_seed=None, verbose=True):
        '''ConvNet4AccelV2 constructor. The job of this method is to build the network as a collection of connected
        layers (in order) in `self.layers`.

        Parameters:
        -----------
        input_shape: tuple. By default: (n_chans, img_y, img_x)
            Shape of a SINGLE input sample (no mini-batch).
        n_kers: tuple.
            Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple rather than just an
            int. The reasoning is that if you wanted to create another Conv2D layer, say with 16 units, n_kers would
            then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple.
            x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST Conv2D layer.
            If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Same format as above.
            Number of hidden units in each dense layer.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple.  Same format as above.
            Pooling extent in the i-th MaxPool2D layer.
        pooling_strides: tuple.  Same format as above.
            Pooling stride in the i-th MaxPool2D layer.
        n_classes: int.
            Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float.
            Global weight scaling to use for all layers with weights
        reg: float.
            Regularization strength
        dropout_rate: float.
            Proportion of units whose activations we "drop" during training when processing each mini-batch.
        r_seed: int or None.
            Random seed for setting weights and bias parameters.
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        NOTE:
        - Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        - (NEW) Remember to define self.dropout_layer_inds as the list of layer indicies that are Dropout layers.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        #creating the layers(Replacing Conv2d and maxpool2d)
        conv_layer = accelerated_layer.Conv2DAccel(0, "conv2daccel", n_kers[0], ker_sz[0], n_chans, wt_scale, 'relu', reg, r_seed, verbose)
        max_layer = accelerated_layer.MaxPool2DAccel(1, "maxpool2d", pooling_sizes[0], pooling_strides[0], 'linear', reg, verbose)
        flat_layer = layer.Flatten(2, 'flatten', verbose)

        max_h = ((h - pooling_sizes[0])/pooling_strides[0]) + 1
        max_w = ((w - pooling_sizes[0])/pooling_strides[0]) + 1
        prev_units = int(n_kers[0] * max_h * max_w) #number of units in previous 2D layer collasped, like in flatten layer
        hidden_layer = layer.Dense(3, 'hidden', dense_interior_units[0], prev_units, wt_scale, 'relu', reg, r_seed, verbose)
        dropout_layer = layer.Dropout(4, 'dropout', rate= dropout_rate, r_seed=r_seed, verbose = verbose)
        output_layer = layer.Dense(5, 'output', n_classes, dense_interior_units[0], wt_scale, 'softmax', reg, r_seed, verbose)

        #creating the neural network

        self.layers = [conv_layer, max_layer, flat_layer, hidden_layer, dropout_layer, output_layer]
        self.wt_layer_inds = [0,3,5]
        self.dropout_layer_inds = [4]

        print("Finished creating network")
