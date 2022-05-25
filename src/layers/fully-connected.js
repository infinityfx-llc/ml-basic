const IllegalArgumentException = require('../exceptions/illegal-argument');
const Sigmoid = require('../functions/sigmoid');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Layer = require('./layer');

module.exports = class FullyConnectedLayer extends Layer {

    static name = 'fully_connected';

    constructor({ input = 2, output = 1, activation = Sigmoid, optimizer = BatchGradientDescent, hyper_parameters = {} } = {}) {
        super(activation, optimizer, hyper_parameters);
        if (input < 1 || output < 1 || !Number.isInteger(input) || !Number.isInteger(output)) throw new IllegalArgumentException('Input and output must be integers greater than 0');

        this.shape = {
            input,
            output
        };
        this.weights = Matrix.random(output, input, -1, 1);
        this.bias = Matrix.random(output, 1, -1, 1);
    }

    clone() {
        const layer = super.clone(new FullyConnectedLayer());
        layer.weights = new Matrix(this.weights);
        layer.bias = new Matrix(this.bias);

        return layer;
    }

    flush() {
        this.weights = Matrix.random(this.shape.output, this.shape.input, -1, 1);
        this.bias = Matrix.random(this.shape.output, 1, -1, 1);
        super.flush();
    }

    shapeInput(input) {
        return this.shapeData(input, this.shape.input);
    }

    propagate(input) {
        input = Matrix.multiply(this.weights, this.shapeInput(input));
        input.add(this.bias);

        return input.transform(this.activation.function);
    }

    backPropagate(input, output, loss, hyper_parameters) {
        output.transform(this.activation.derivative).flat();
        loss.flat();

        this.optimizer.useParameters(hyper_parameters); //make temp
        const gradient = this.optimizer.step(Matrix.product(output, loss));

        if (gradient) {
            this.bias.add(gradient);

            this.weights.add(gradient.multiply(Matrix.flat(input).transpose()));
        }

        return Matrix.multiply(Matrix.transpose(this.weights), loss);
    }

    cross(b) {
        this.weights = Matrix.add(this.weights, b.weights).scale(0.5);
        this.bias = Matrix.add(this.bias, b.bias).scale(0.5);

        return this;
    }

    mutate({ mutation_probability = 0.2, mutation_constant = 0.5 } = {}) {
        const w_hat = Matrix.random(this.shape.output, this.shape.input, -mutation_constant, mutation_constant)
            .transform(val => Math.pow(val, Math.round(Math.sqrt(1 / mutation_probability))));
        this.weights.add(w_hat);

        const b_hat = Matrix.random(this.shape.output, 1, -mutation_constant, mutation_constant)
            .transform(val => Math.pow(val, Math.round(Math.sqrt(1 / mutation_probability))));
        this.bias.add(b_hat);
    }

    static deserialize(data) {
        return super.deserialize(data, new FullyConnectedLayer());
    }

}