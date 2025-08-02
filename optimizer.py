'''optimizer.py
Algorithms to optimize the weights during gradient descent / backprop
Samuel Atilano
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import numpy as np


class Optimizer:
    def __init__(self):
        self.wts = None
        self.d_wts = None

    def prepare(self, wts, d_wts):
        '''Stores weights and their gradient before an update step is performed.
        '''
        self.wts = wts
        self.d_wts = d_wts

    def update_weights(self):
        pass

    @staticmethod
    def create_optimizer(name, *args, **kwargs):
        '''Factory method that takes in a string, and returns a new object of the
        desired type. Called via Optimizer.create_optimizer().
        '''
        if name.lower() == 'sgd':
            return SGD(**kwargs)
        elif name.lower() == 'sgd_momentum' or name.lower() == 'sgd_m' or name.lower() == 'sgdm':
            return SGD_Momentum(**kwargs)
        elif name.lower() == 'adam':
            return Adam(**kwargs)
        elif name.lower() == 'adamw':
            print("creating adamw")
            return AdamW(*args, **kwargs)
        else:
            raise ValueError('Unknown optimizer name!')


class SGD(Optimizer):
    '''Update weights using Stochastic Gradient Descent (SGD) update rule.
    '''
    def __init__(self, lr=0.1):
        '''SGD optimizer constructor

        Parameters:
        -----------
        lr: float > 0. Learning rate.
        '''
        self.lr = lr

    def update_weights(self):
        '''Updates the weights according to SGD and returns a deep COPY of the
        updated weights for this time step.

        Returns:
        -----------
        The updated weights for this time step.
        '''
        
        wt = self.wts - (self.lr * self.d_wts)
        self.wts = wt
        return self.wts.copy()


class SGD_Momentum(Optimizer):
    '''Update weights using Stochastic Gradient Descent (SGD) with momentum
    update rule.
    '''
    def __init__(self, lr=0.001, m=0.9):
        '''SGD-M optimizer constructor

        Parameters:
        -----------
        lr: float > 0. Learning rate.
        m: float 0 < m < 1. Amount of momentum from gradient on last time step.
        '''
        self.lr = lr
        self.m = m
        self.velocity = None

    def update_weights(self):
        '''Updates the weights according to SGD with momentum and returns a
        deep COPY of the updated weights for this time step.

        Returns:
        -----------
        The updated weights for this time step.
        '''

        if self.velocity is None:
            self.velocity = np.zeros(shape=self.wts.shape)

        #Calculating the weights on this time step
        vt = (self.m * self.velocity) - (self.lr * self.d_wts)
        wt = self.wts + vt

        #Updating weights and velocity
        self.velocity = vt
        self.wts = wt

        return self.wts.copy()


class Adam(Optimizer):
    '''Update weights using the Adam update rule.
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=0):
        '''Adam optimizer constructor

        Parameters:
        -----------
        lr: float > 0. Learning rate.
        beta1: float. 0 < beta1 < 1. Amount of momentum from gradient on last time step.
        beta2: float. 0 < beta2 < 1. Amount of momentum from gradient on last time step.
        eps: float. Small number to prevent division by 0.
        t: int. Records the current time step: 0, 1, 2, ....
        '''
        #Assigning values to instance variables
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = t

        self.v = None
        self.p = None

    def update_weights(self):
        '''Updates the weights according to Adam and returns a
        deep COPY of the updated weights for this time step.

        Returns:
        -----------
        The updated weights for this time step.
        '''
        
        #Initializing v and p
        if self.v is None and self.p is None:
            self.v = np.zeros(shape=self.wts.shape)
            self.p = np.zeros(shape=self.wts.shape)

        self.t += 1

        #Calculating the weights
        vt = (self.beta1 * self.v) + ((1 - self.beta1) * self.d_wts)
        pt = (self.beta2 * self.p) + ((1 - self.beta2) * (self.d_wts**2))

        self.v = vt
        self.p = pt 

        vc = vt / (1 - (self.beta1 ** self.t))
        pc = pt / (1 - (self.beta2 ** self.t))

        wt = self.wts - ((self.lr * vc) / (np.sqrt(pc) + self.eps))

        #Weight Update
        self.wts = wt

        return self.wts.copy()

class AdamW(Optimizer):
    '''Update weights using the AdamW update rule.
    '''
    def __init__(self, reg=None, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=0):
        '''AdamW optimizer constructor

        Parameters:
        -----------
        lr: float > 0. Learning rate.
        reg: float >= 0. Regularization strength (NEW!)
        beta1: float. 0 < beta1 < 1. Amount of momentum from gradient on last time step.
        beta2: float. 0 < beta2 < 1. Amount of momentum from gradient on last time step.
        eps: float. Small number to prevent division by 0.
        t: int. Records the current time step: 0, 1, 2, ....
        '''
        if reg is None:
            print('FYI: Using AdamW without any reg.')

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = t

        self.v = None
        self.p = None

        self.reg = reg #Could be none

    def update_weights(self):
        '''Updates the weights according to Adam and returns a
        deep COPY of the updated weights for this time step.

        Returns:
        -----------
        The updated weights for this time step.
        '''
        
        #Initializing v and p
        if self.v is None and self.p is None:
            self.v = np.zeros(shape=self.wts.shape)
            self.p = np.zeros(shape=self.wts.shape)

        
        #Calculating the weights
        self.t += 1
        vt = (self.beta1 * self.v) + ((1 - self.beta1) * self.d_wts)
        pt = (self.beta2 * self.p) + ((1 - self.beta2) * (self.d_wts**2))

        self.v = vt
        self.p = pt 

        vc = vt / (1 - (self.beta1 ** self.t))
        pc = pt / (1 - (self.beta2 ** self.t))

        wt = (self.wts - (self.lr * vc / (np.sqrt(pc) + self.eps))) - (self.lr * (self.reg * self.wts))

        #Weight Update
        self.wts = wt

        return self.wts.copy()

#Test Cases given to me
def test_sgd():
    rng = np.random.default_rng(0)

    wts = np.arange(-3, 3, dtype=np.float64)
    d_wts = rng.standard_normal(len(wts))

    optimizer = SGD()
    optimizer.prepare(wts, d_wts)

    new_wts_1 = optimizer.update_weights()
    new_wts_2 = optimizer.update_weights()

    print(f'SGD: Wts after 1 iter {new_wts_1}')
    print(f'SGD: Wts after 2 iter {new_wts_2}')


def test_sgd_m():
    rng = np.random.default_rng(0)

    wts = rng.standard_normal(3, 4)
    d_wts = rng.standard_normal(3, 4)

    optimizer = SGD_Momentum(lr=0.1, m=0.6)
    optimizer.prepare(wts, d_wts)

    new_wts_1 = optimizer.update_weights()
    new_wts_2 = optimizer.update_weights()

    print(f'SGD M: Wts after 1 iter\n{new_wts_1}')
    print(f'SGD N: Wts after 2 iter\n{new_wts_2}')


def test_adam():
    rng = np.random.default_rng(0)

    wts = rng.standard_normal(3, 4)
    d_wts = rng.standard_normal(3, 4)

    optimizer = Adam(lr=0.1)
    optimizer.prepare(wts, d_wts)

    new_wts_1 = optimizer.update_weights()
    new_wts_2 = optimizer.update_weights()
    new_wts_3 = optimizer.update_weights()

    print(f'Adam: Wts after 1 iter\n{new_wts_1}')
    print(f'Adam: Wts after 2 iter\n{new_wts_2}')
    print(f'Adam: Wts after 3 iter\n{new_wts_3}')


if __name__ == '__main__':
    # test_sgd()
    # test_sgd_m()
    test_adam()