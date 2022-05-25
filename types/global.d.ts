export interface HyperParameters {
    learning_rate?: number;
    batch_size?: number;
    beta?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
}

export type Layers = 'fully_connected' | 'convolutional' | 'max_pooling';

export type Activation = 'sigmoid' | 'tanh' | 'elu' | 'parametric-relu' | 'softplus';

export type Optimizer = 'gradient_descent' | 'batch_gradient_descent' | 'rms_prop' | 'adam';

export type Loss = 'squared_loss';