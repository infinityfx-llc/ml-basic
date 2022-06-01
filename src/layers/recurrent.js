const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Sigmoid = require('../functions/sigmoid');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Layer = require('./layer');

module.exports = class RecurrentLayer extends Layer {

    static name = 'recurrent';

    constructor({ input = [2, 2], output = 1, offset = 0, activation = Sigmoid, optimizer = BatchGradientDescent, hyper_parameters = {} } = {}) {
        super(activation, optimizer, hyper_parameters);
        if (output < 1 || !Number.isInteger(output)) throw new IllegalArgumentException('`output` must be an integer greater than 0');
        if (!Array.isArray(input) || input[0] < 1 || input[1] < 1 || !Number.isInteger(input[0]) || !Number.isInteger(input[1])) throw new IllegalArgumentException('`input` must be an Array of integers greater than 0');

        this.shape = {
            input,
            output: [
                input[0],
                output
            ]
        };
        this.weights = Matrix.random(input[0], input[0], -1, 1);
        this.bias = Matrix.random(input[0], 1, -1, 1);
        this.state = new Matrix(input[0], 1);

        this.offset = offset;
    }

    shapeInput(input) {
        return this.shapeData(input, this.shape.input[0]);
    }

    propagate(input) {
        if (input.length !== this.shape.input[1]) throw new IllegalArgumentException('`input` length must be equal to layer input size');

        let output = [];

        for (let i = 0; i < this.shape.input[1] + this.offset; i++) {

            if (i < this.shape.input[1]) {
                input = Matrix.multiply(this.weights, this.shapeInput(input));
                input.add(this.bias);
            }

            const i_hat = Matrix.multiply(this.weights, this.state);
            i_hat.add(this.bias);
            input.add(i_hat).transform(this.activation.function);

            this.state = input.clone(); //maybe don't need to clone

            if (i < this.shape.input[1] - this.shape.output[1] + this.offset) continue;

            output.push(input);
        }
        
        this.state.zeros();

        return output;
    }

    backPropagate(input, output, loss, hyper_parameters) {
        throw new Exception('Not yet implemented');
    }

    static deserialize(data) {
        return super.deserialize(data, new RecurrentLayer());
    }

}