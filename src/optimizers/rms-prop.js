const IllegalArgumentException = require('../exceptions/illegal-argument');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('./batch-gradient-descent');

module.exports = class RMSProp extends BatchGradientDescent {

    static name = 'rms_prop';

    constructor({ learning_rate = 0.01, batch_size = 8, beta = 0.9, epsilon = 1e-8 } = {}) {
        super({ learning_rate, batch_size });
        if (epsilon <= 0) throw new IllegalArgumentException('Epsilon must be a number greater than 0');

        this.beta = beta;
        this.epsilon = epsilon;

        this.avg = null;
    }

    flush() {
        this.avg = null;
        super.flush();
    }

    useParameters({ learning_rate = this.learning_rate, batch_size = this.batch_size, beta = this.beta, epsilon = this.epsilon } = {}) {
        super.useParameters({ learning_rate, batch_size });
        if (epsilon <= 0) throw new IllegalArgumentException('Epsilon must be a number greater than 0');
        
        this.beta = beta;
        this.epsilon = epsilon;
    }

    step(gradient) {
        if (!this.avg) {
            this.avg = new Matrix(gradient).ones();
        }

        if (this.i % this.batch_size !== 0) return this.i++, null;

        this.avg.scale(this.beta).add(Matrix.scale(Matrix.transform(gradient, val => Math.pow(val, 2)), 1 - this.beta));

        gradient = gradient.product(Matrix.transform(this.avg, val => Math.sqrt(val)).add(this.epsilon).transform(val => 1 / val));

        return super.step(gradient);
    }

    static deserialize(data) {
        return super.deserialize(data, new RMSProp());
    }

};