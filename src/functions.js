const CrossEntropyLoss = require('./functions/cross-entropy-loss');
const SquaredLoss = require('./functions/squared-loss');

module.exports = {

    loss: {
        [SquaredLoss.name]: SquaredLoss,
        [CrossEntropyLoss.name]: CrossEntropyLoss
    }
    
};