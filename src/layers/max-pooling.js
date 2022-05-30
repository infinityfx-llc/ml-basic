const Sigmoid = require('../functions/sigmoid');
const PoolingLayer = require('./pooling');

module.exports = class MaxPoolingLayer extends PoolingLayer {

    static name = 'max_pooling';

    constructor({ input = [8, 8], size = [2, 2], stride = 2, activation = Sigmoid } = {}) {
        super({ input, size, stride, activation });
    }

    clone() {
        return super.clone(new MaxPoolingLayer());
    }

    pool(value, _, max = -Number.MAX_VALUE) {
        return value > max ? value : max;
    }

    unpool(value, index, max = -Number.MAX_VALUE, indices = [0]) {
        if (value > max) {
            indices[0] = index;
            max = value;
        }
        
        return [max, indices];
    }

    static deserialize(data) {
        return super.deserialize(data, new MaxPoolingLayer());
    }

}