import Sigmoid from './functions/sigmoid';
import TanH from './functions/tanh';
import ParametricRelu from './functions/parametric-relu';
import Elu from './functions/elu';
import Softplus from './functions/softplus';
import SquaredLoss from './functions/squared-loss';
import Optimizer from './optimizers/optimizer';
import BatchGradientDescent from './optimizers/batch-gradient-descent';
import Adam from './optimizers/adam';

export const TYPES = {
    [Sigmoid.name]: Sigmoid,
    [TanH.name]: TanH,
    [ParametricRelu.name]: ParametricRelu,
    [Elu.name]: Elu,
    [Softplus.name]: Softplus,
    [SquaredLoss.name]: SquaredLoss,
    [Optimizer.name]: Optimizer,
    [BatchGradientDescent.name]: BatchGradientDescent,
    [Adam.name]: Adam
};