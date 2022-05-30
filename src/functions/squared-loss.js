const Matrix = require('../math/matrix');
const Loss = require('./loss');

module.exports = class SquaredLoss extends Loss {

    static name = 'squared_loss';

    static mean(output, target) {
        return Matrix.sub(target, output).transform((val) => val * val).sum() / target.entries.length;
    }

    static derivative(output, target) {
        // return Matrix.sub(target, output);
        return Matrix.sub(target, output).transform(val => -2 * val / target.entries.length);
    }

};