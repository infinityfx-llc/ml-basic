const IllegalArgumentException = require('../exceptions/illegal-argument');
const Optimizer = require('./optimizer');
const Matrix = require('../math/matrix');

module.exports = class BatchGradientDescent extends Optimizer {

    static name = 'batch_gradient_descent';

    constructor({ learning_rate = 0.1, batch_size = 8 } = {}) {
        super({ learning_rate });
        if (batch_size <= 0 || !Number.isInteger(batch_size)) throw new IllegalArgumentException('Batch size must be an Integer greater than 0');

        this.i = 1;
        this.batch_size = batch_size;
        this.aggregate = null;
    }

    useParameters({ learning_rate = this.learning_rate, batch_size = this.batch_size } = {}) {
        super.useParameters({ learning_rate });
        if (batch_size <= 0 || !Number.isInteger(batch_size)) throw new IllegalArgumentException('Batch size must be an Integer greater than 0');
        
        this.batch_size = batch_size;
    }

    step(layer, input, gradient) {
        this.aggregate ? this.aggregate.add(gradient) : this.aggregate = gradient;
        
        if (this.i % this.batch_size === 0) {
            super.step(layer, input, Matrix.scale(this.aggregate, 1 / this.batch_size));
            this.aggregate.zeros();
        }

        this.i++;
    }

    static deserialize(data) {
        return super.deserialize(data, new BatchGradientDescent());
    }

};