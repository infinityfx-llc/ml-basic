const IllegalArgumentException = require('../exceptions/illegal-argument');
const Matrix = require('../math/matrix');

module.exports = class Optimizer {

    static name = 'gradient_descent';

    constructor({ learning_rate = 0.1 } = {}) {
        if (learning_rate <= 0) throw new IllegalArgumentException('Learning rate must be a number greater than 0');

        this.t = 1;
        this.learning_rate = learning_rate;
    }

    flush() {
        this.t = 1;
    }

    useParameters({ learning_rate = this.learning_rate } = {}) {
        if (learning_rate <= 0) throw new IllegalArgumentException('Learning rate must be a number greater than 0');

        this.learning_rate = learning_rate;
    }

    step(layer, input, gradient) {
        this.t++;

        gradient.scale(this.learning_rate);
        layer.bias.add(gradient);

        layer.weights.add(gradient.multiply(Matrix.transpose(input)));
    }

    serialize() {
        return Object.entries(this).reduce((map, [key, val]) => {
            if (val instanceof Matrix) val = val.serialize();

            return { ...map, [key]: val };
        }, { name: this.__proto__.constructor.name });
    }

    static deserialize(data, optimizer = new Optimizer()) {
        Object.entries(data).forEach(([key, val]) => {
            if (key === 'aggregate' || key === 'avg') return optimizer[key] = val ? Matrix.deserialize(val) : val;

            optimizer[key] = val;
        });

        return optimizer;
    }

};