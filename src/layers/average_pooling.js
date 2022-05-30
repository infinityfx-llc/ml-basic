const Sigmoid = require('../functions/sigmoid');
const PoolingLayer = require('./pooling');

module.exports = class AveragePooling extends PoolingLayer {

    static name = 'average_pooling';

    constructor({ input = [8, 8], size = [2, 2], stride = 2, activation = Sigmoid } = {}) {
        super({ input, size, stride, activation });
    }

    clone() {
        return super.clone(new AveragePooling());
    }

    pool(value, length, avg = 0) {
        return avg + value / length;
    }

    unpool(_, index, __, indices = []) {
        indices.push(index);
        
        return [null, indices];
    }

    static deserialize(data) {
        return super.deserialize(data, new AveragePooling());
    }

}