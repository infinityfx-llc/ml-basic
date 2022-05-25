const IllegalArgumentException = require('../exceptions/illegal-argument');
const Optimizer = require('./optimizer');
const Matrix = require('../math/matrix');

module.exports = class BatchGradientDescent extends Optimizer {

    static name = 'batch_gradient_descent';

    constructor({ learning_rate = 0.1, batch_size = 8 } = {}) {
        super({ learning_rate });
        if (batch_size <= 0 || !Number.isInteger(batch_size)) throw new IllegalArgumentException('Batch size must be an Integer greater than 0');

        this.i = 0;
        this.batch_size = batch_size;
        this.aggregate = null;
    }

    flush() {
        this.i = 0;
        this.aggregate = null;
        super.flush();
    }

    useParameters({ learning_rate = this.learning_rate, batch_size = this.batch_size } = {}) {
        super.useParameters({ learning_rate });
        if (batch_size <= 0 || !Number.isInteger(batch_size)) throw new IllegalArgumentException('Batch size must be an Integer greater than 0');
        
        this.batch_size = batch_size;
    }

    step(gradient) {
        this.aggregate ? this.aggregate.add(gradient) : this.aggregate = gradient;
        this.i++;

        if (this.i % this.batch_size === 0) {
            gradient = super.step(Matrix.scale(this.aggregate, 1 / this.batch_size));
            this.aggregate.zeros();
            return gradient;
        }

        return null;
    }

    static deserialize(data, optimizer = new BatchGradientDescent()) {
        return super.deserialize(data, optimizer);
    }

};