const IllegalArgumentException = require('../exceptions/illegal-argument');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('./batch-gradient-descent');

module.exports = class Adam extends BatchGradientDescent {

    static name = 'adam';

    constructor({ learning_rate = 0.01, batch_size = 8, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = {}) {
        super({ learning_rate, batch_size });
        if (epsilon <= 0) throw new IllegalArgumentException('Epsilon must be a number greater than 0');

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        this.m = null;
        this.v = null;
    }

    flush() {
        this.m = null;
        super.flush();
    }

    useParameters({ learning_rate = this.learning_rate, batch_size = this.batch_size, beta1 = this.beta1, beta2 = this.beta2, epsilon = this.epsilon } = {}) {
        super.useParameters({ learning_rate, batch_size });
        if (epsilon <= 0) throw new IllegalArgumentException('Epsilon must be a number greater than 0');
        
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    step(gradient) {
        if (!this.m) {
            this.m = new Matrix(gradient).zeros();
            this.v = new Matrix(gradient).zeros();
        }

        if (this.i % this.batch_size !== 0) return this.i++, null;

        this.m = Matrix.scale(this.m, this.beta1).add(Matrix.scale(gradient, 1 - this.beta1));
        this.v = Matrix.scale(this.v, this.beta2).add(Matrix.transform(gradient, val => val * val).scale(1 - this.beta2));

        let m_hat = Matrix.scale(this.m, 1 / (1 - Math.pow(this.beta1, this.t))),
            v_hat = Matrix.scale(this.v, 1 / (1 - Math.pow(this.beta2, this.t)));

        gradient = m_hat.product(v_hat.transform(val => Math.sqrt(val)).add(this.epsilon).transform(val => 1 / val)).scale(-1);

        return super.step(gradient);
    }

    static deserialize(data) {
        return super.deserialize(data, new Adam());
    }

};