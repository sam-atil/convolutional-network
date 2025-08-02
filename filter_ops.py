'''filter_ops.py
Implements the convolution and max pooling operations.
Applied to images and other data represented as an ndarray.
Samuel Atilano
CS343: Neural Networks
Project 3: Convolutional neural networks
'''
import numpy as np


def conv2_gray(img, kers, verbose=True):
    '''Performs 2D convolution operation on GRAYSCALE images `img` by using kernels `kers`.

    Parameters:
    -----------
    img: ndarray. Grayscale input image to be filtered. shape=(height img_y (px), width img_x (px))
    kers: ndarray. Convolution kernels. shape=(Num kers, ker_sz (px), ker_sz (px))
        NOTE: Kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. debugging tool to print shapes of variables as the program runs

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all the `kers`. shape=(Num kers, img_y, img_x)
    '''

    #Instance Variables
    img_y, img_x = img.shape
    n_kers, ker_y, ker_x = kers.shape

    #Shape Debugging
    if verbose:
        print(f'img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_y={ker_y}, ker_x={ker_x}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return

    kers_flip = np.flip(kers) #Flips across horizontal and vertical axes
    
    #Determines the Pad Size and creates the Padded Image
    pad_sz = np.ceil((ker_y - 1) / 2).astype(int)
    padded_img = np.pad(img, pad_width=pad_sz, constant_values=0)
    
    #Creates the output img, dimensions of y and x match the input dimensions
    output_img = np.zeros(shape=(n_kers, img_y, img_x))

    #Shape Debugging
    if verbose:
        print(f'Padding Size: {pad_sz}')
        print(f'Padded Image: {padded_img.shape}')
        print(f'output image shape: {output_img.shape}')

    #Convolution Process
    for k in range(n_kers): #Loops through k kernals/filters
        for i in range(img_y): #Loops through x dimension of input image
            for j in range(img_x): #loops through y dimension of input image
                #Creates the window box range for the padded image
                y_range = i + ker_y
                x_range = j + ker_x

                #Creates the window box from the padded image
                window = padded_img[i: y_range, j: x_range]

                #Applies point-wise multiplication between window and kth kernal
                #Sums the result to create a scalar value
                output_img[k, i, j] = np.sum(window * kers_flip[k])
        
    return output_img


def conv2(img, kers, verbose=True):
    '''Does a 2D convolution operation on COLOR or grayscale `img` using kernels `kers`.
    Key difference between conv2 and conv2_gray is that the img variable now contains a 'chan'
    dimension for the color channels.

    Parameters:
    -----------
    img: ndarray. Input image to be filtered. shape=(N_CHANS, height img_y (px), width img_x (px))
        where n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(Num filters, ker_sz (px), ker_sz (px))
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. Print Statement Debugging Tool 

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all `kers`. shape=(Num filters, N_CHANS, img_y, img_x)

    '''

    #Instance Variables
    n_chans, img_y, img_x = img.shape
    n_kers, ker_y, ker_x = kers.shape

    #Shape Debugging
    if verbose:
        print(f'n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_y={ker_y}, ker_x={ker_x}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return

    kers_flip = np.flip(kers, axis = (1,2)) #Flips across horizontal and vertical axes
    
    #Determines the Pad Size and creates the Padded Image
    pad_sz = np.ceil((ker_y - 1) / 2).astype(int)
    padded_img = np.pad(img, ((0,0), (pad_sz, pad_sz), (pad_sz, pad_sz)), constant_values=0)
    
    output_img = np.zeros(shape=(n_kers, n_chans, img_y, img_x))

    #Shape Debugging
    if verbose:
        print(f'Kernal Shape: {kers_flip.shape}') 
        print(f'Padding Size: {pad_sz}')
        print(f'Padded Image: {padded_img.shape}')
        print(f'output image shape: {output_img.shape}')

    #Convolution Process
    for k in range(n_kers): #Loops through k Kernals/Filters
        for i in range(img_y): #Loops through columns of Input Image
            for j in range(img_x): #Loops through rows of Input Image

                #Sets up the Range for filter window
                y_range = i + ker_y
                x_range = j + ker_x

                #Creates the window from the padded image
                window = padded_img[:, i: y_range, j: x_range]

                #Applies the Point-Wise multiplication between the window and kth kernal
                #Then sums the result to create a scalar value
                output_img[k, :, i, j] = np.sum((window * kers_flip[k]),axis = (1,2)) #I am incorrectly applying all 4 kernals across the 3 color channels here

    return output_img


def conv2nn(imgs, kers, bias, verbose=True):
    '''General 2D convolution operation.

    Parameters:
    -----------
    imgs: ndarray. Input IMAGES to be filtered. shape=(BATCH_SZ, n_chans, img_y, img_x)
        where batch_sz is the number of images in the mini-batch
        n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(n_kers, N_CHANS, ker_sz, ker_sz)
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    bias: ndarray. Bias term used in the neural network layer. Shape=(n_kers,)
        i.e. there is a single bias per filter. Convolution by the c-th filter gets the c-th
        bias term added to it.
    verbose: bool. Debugging tool.

    What's new (vs conv2):
    -----------
    - Multiple images (mini-batch support)
    - Kernels now have a color channel dimension too
    - Collapse (sum) over color channels when computing the returned output images
    - A bias term

    Returns:
    -----------
    output: ndarray. `imgs` filtered with all `kers`. shape=(BATCH_SZ, n_kers, img_y, img_x)
    '''
    
    #Instance Variables
    batch_sz, n_chans, img_y, img_x = imgs.shape
    n_kers, n_ker_chans, ker_y, ker_x = kers.shape

    #Debugging print statements
    if verbose:
        print(f'batch_sz={batch_sz}, n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, n_ker_chans={n_ker_chans}, ker_y={ker_y}, ker_x={ker_x}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return

    if n_chans != n_ker_chans:
        print('Number of kernel channels doesnt match input num channels!')
        return

    kers_flip = np.flip(kers, axis = (2,3)) #Flips across horizontal and vertical axes
 
    #Determines the Pad Size and creates the Padded Image
    pad_sz = np.ceil((ker_y - 1) / 2).astype(int)
    padded_imgs = np.pad(imgs, ((0,0), (0,0), (pad_sz, pad_sz), (pad_sz, pad_sz)), constant_values=0)
    
    #Creates output image of shape (Batch_sz, n_kers, img_y, img_x)
    output_img = np.zeros(shape=(batch_sz, n_kers, img_y, img_x))

    #Debugging statements
    if verbose:
        print(f'Kernal Shape: {kers_flip.shape}')
        print(f'Padding Size: {pad_sz}')
        print(f'Padded Image: {padded_imgs.shape}')
        print(f'output image shape: {output_img.shape}')

    #Convolution Process
    for k in range(n_kers): #Loops through k Kernals/Filters
        for i in range(img_y): #Loops through columns of Input Image
            for j in range(img_x): #Loops through rows of Input Image

                #Sets up the Range for filter window
                y_range = i + ker_y
                x_range = j + ker_x

                #Creates the window from the padded image
                window = padded_imgs[:, :, i: y_range, j: x_range] #of Shape (batch_sz, n_chans, ker_y, ker_x)

                #Applies the Point-Wise multiplication between the window and kth kernal
                #Then sums the result to create a scalar value
                output_img[:, k, i, j] = np.sum((window * kers_flip[k]), axis = (1,2,3))

        #Adds the kth bias term
        output_img[:, k, :, :] += bias[k]

    return output_img



#Helper Function
def get_pooling_out_shape(img_dim, pool_size, strides):
    '''Computes the size of the output of a max pooling operation along one spatial dimension.

    Parameters:
    -----------
    img_dim: int. Either img_y or img_x
    pool_size: int. Size of pooling window in one dimension: either x or y (assumed the same).
    strides: int. Size of stride when the max pooling window moves from one position to another.

    Returns:
    -----------
    int. The size in pixels of the output of the image after max pooling is applied, in the dimension
        img_dim.
    '''
    
    floor = int(((img_dim - pool_size) / strides) + 1)
    return floor


def max_pool(inputs, pool_size=2, strides=1, verbose=True):
    ''' Does max pooling on inputs. Works on single grayscale images, so somewhat comparable to
    `conv2_gray`.

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(height img_y, width img_x)
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that? :)

    NOTE: There is no padding in the max-pooling operation.

    Hints:
    -----------
    - You should be able to heavily leverage the structure of your conv2_gray code here
    - Instead of defining a kernel, indexing strategically may be helpful
    - You may need to keep track of and update indices for both the input and output images
    - Overall, this should be a simpler implementation than `conv2_gray`
    '''

    #Instance Variables
    img_y, img_x = inputs.shape

    #Debugging Statement
    if verbose:
        print(f'Img_y: {img_y}, Img_x: {img_x}')

    #calculates the pooling output shape of the x and y dimensions
    out_y = get_pooling_out_shape(img_y, pool_size, strides)
    out_x = get_pooling_out_shape(img_x, pool_size, strides)

    #Debugging Statement
    if verbose:
        print(f'Out_Y: {out_y}')
        print(f'Out_Y: {out_x}')

    #Empty Image
    output_img = np.zeros(shape=(out_y, out_x))

    #Max-pooling operation
    for i in range(out_y): 
        for j in range(out_x):
            #Sets up the pooling range along the x and y dimensions    
            input_i = i * strides
            input_j = j * strides
            y_range = input_i + pool_size
            x_range = input_j + pool_size

            max_num = np.max(inputs[input_i: y_range, input_j: x_range]) #Scalar value
            output_img[i, j] = max_num #Assigns the largest number to the output img
    
    return output_img


def max_poolnn(inputs, pool_size=2, strides=1, verbose=True):
    ''' Max pooling implementation for a MaxPool2D layer of a neural network

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(mini_batch_sz, n_chans, height img_y, width img_x)
        where n_chans is 1 for grayscale images and 3 for RGB color images
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(mini_batch_sz, n_chans, out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that?)

    What's new (vs max_pool):
    -----------
    - Multiple images (mini-batch support)
    - Images now have a color channel dimension too

    '''

    #Instance Variables
    mini_batch_sz, n_chans, img_y, img_x = inputs.shape

    #Debugging Statements
    if verbose:
        print(f'Img_y: {img_y}, Img_x: {img_x}')
        print(f'Mini-batch: {mini_batch_sz}')
        print(f'N_Chans: {n_chans}')

    #pooling output for x and y dimensions
    out_y = get_pooling_out_shape(img_y, pool_size, strides)
    out_x = get_pooling_out_shape(img_x, pool_size, strides)

    #Printing Statements
    if verbose:
        print(f'Out_Y: {out_y}')
        print(f'Out_X: {out_x}')

    #Empty img
    output_img = np.zeros(shape=(mini_batch_sz, n_chans, out_y, out_x))

    for i in range(out_y): 
        for j in range(out_x):
            #Setting up the x and y pool range    
            input_i = i * strides
            input_j = j * strides
            y_range = input_i + pool_size
            x_range = input_j + pool_size

            max_num = np.max(inputs[:, :, input_i: y_range, input_j: x_range], axis = (2,3)) #Scalar value
            output_img[:, :, i, j] = max_num
    
    return output_img