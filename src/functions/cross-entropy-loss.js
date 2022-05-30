const Matrix = require('../math/matrix');
const Loss = require('./loss');

module.exports = class CrossEntropyLoss extends Loss {

    static name = 'cross_entropy';

    static mean(output, target) {
        return -Matrix.transform(output, val => Math.log(val)).product(target).sum();
    }

    static derivative(output, target) {
        return Matrix.sub(output, target);
    }

};