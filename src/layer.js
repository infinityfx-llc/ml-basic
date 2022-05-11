import Activator from './functions/activator';
import Sigmoid from './functions/sigmoid';
import IllegalArgumentException from './exceptions/illegal-argument';
import Matrix from './math/matrix';
import Optimizer from './optimizers/optimizer';
import BatchGradientDescent from './optimizers/batch-gradient-descent';
import { TYPES } from './types';

export default class Layer {

    constructor(input = 2, output = 1, activation = Sigmoid, optimizer = BatchGradientDescent, hyper_parameters = {}) {
        if (input < 1 || output < 1) throw new IllegalArgumentException('Input and output must be numbers greater than 0');
        if (!(activation.prototype instanceof Activator)) throw new IllegalArgumentException('Activation must be of type Activator');
        if (!(optimizer.prototype instanceof Optimizer)) throw new IllegalArgumentException('Optimizer must be of type Optimizer');

        this.size = {
            input,
            output
        };
        this.weights = Matrix.random(output, input, -1, 1);
        this.bias = Matrix.random(output, 1, -1, 1);

        this.activation = activation;
        this.optimizer = new optimizer(hyper_parameters);
    }

    equals(layer) {
        return this.size.input === layer.size.input &&
            this.size.output === layer.size.output &&
            this.activation === layer.activation &&
            this.optimizer.__proto__.constructor === layer.optimizer.__proto__.constructor;
    }

    flush() {
        this.weights = Matrix.random(this.size.output, this.size.input, -1, 1);
        this.bias = Matrix.random(this.size.output, 1, -1, 1);
        this.optimizer.flush();
    }

    propagate(input) {
        input = Matrix.multiply(this.weights, input);
        input.add(this.bias);

        return input.transform(this.activation.function);
    }

    backPropagate(input, output, loss, hyper_parameters) {
        output.transform(this.activation.derivative);

        this.optimizer.useParameters(hyper_parameters); //make temp
        this.optimizer.step(this, input, Matrix.product(output, loss));

        return Matrix.multiply(Matrix.transpose(this.weights), loss);
    }

    static cross(a, b) {
        if (!(a instanceof Layer && b instanceof Layer)) throw new IllegalArgumentException('A and b must be an instance of Layer');
        if (!a.equals(b)) throw new IllegalArgumentException('A must equal b');

        const result = new Layer(a.size.input, a.size.output, a.activation);
        result.weights = Matrix.add(a.weights, b.weights).scale(0.5);
        result.bias = Matrix.add(a.bias, b.bias).scale(0.5);

        return result;
    }

    mutate({ mutation_probability = 0.2, mutation_constant = 0.5 } = {}) {
        const w_hat = Matrix.random(this.size.output, this.size.input, -mutation_constant, mutation_constant)
            .transform(val => Math.pow(val, Math.round(Math.sqrt(1 / mutation_probability))));
        this.weights.add(w_hat);

        const b_hat = Matrix.random(this.size.output, 1, -mutation_constant, mutation_constant)
            .transform(val => Math.pow(val, Math.round(Math.sqrt(1 / mutation_probability))));
        this.bias.add(b_hat);
    }

    serialize() {
        return Object.entries(this).reduce((map, [key, val]) => {
            if (val.prototype instanceof Activator || val instanceof Optimizer || val instanceof Matrix) val = val.serialize();

            return { ...map, [key]: val };
        }, {});
    }

    static deserialize(data) {
        const layer = new Layer();

        Object.entries(data).forEach(([key, val]) => {
            if (key === 'activation') return layer[key] = TYPES[val];
            if (key === 'optimizer') return layer[key] = TYPES[val.name].deserialize(val);
            if (key === 'weights' || key === 'bias') return layer[key] = Matrix.deserialize(val);

            layer[key] = val;
        });

        return layer;
    }

}