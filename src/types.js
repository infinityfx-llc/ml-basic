const Sigmoid = require('./functions/sigmoid');
const TanH = require('./functions/tanh');
const ParametricRelu = require('./functions/parametric-relu');
const Elu = require('./functions/elu');
const Softplus = require('./functions/softplus');
const SquaredLoss = require('./functions/squared-loss');
const Optimizer = require('./optimizers/optimizer');
const BatchGradientDescent = require('./optimizers/batch-gradient-descent');
const Adam = require('./optimizers/adam');
const RMSProp = require('./optimizers/rms-prop');
const CrossEntropyLoss = require('./functions/cross-entropy-loss');

module.exports.TYPES = {
    [Sigmoid.name]: Sigmoid,
    [TanH.name]: TanH,
    [ParametricRelu.name]: ParametricRelu,
    [Elu.name]: Elu,
    [Softplus.name]: Softplus,
    [SquaredLoss.name]: SquaredLoss,
    [CrossEntropyLoss.name]: CrossEntropyLoss,
    [Optimizer.name]: Optimizer,
    [BatchGradientDescent.name]: BatchGradientDescent,
    [RMSProp.name]: RMSProp,
    [Adam.name]: Adam
};