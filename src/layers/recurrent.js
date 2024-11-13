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

    // TODO: dynamic input size propagation
    propagate(input) {
        if (input.length !== this.shape.input[1]) throw new IllegalArgumentException('`input` length must be equal to layer input size');

        this.inputs = [];
        this.outputs = [];

        for (let i = 0; i < this.shape.input[1] + this.offset; i++) {

            let partial_input;
            if (i < this.shape.input[1]) {
                this.inputs.push(this.shapeInput(input[i]));
                partial_input = Matrix.multiply(this.weights, this.inputs[this.inputs.length - 1]);
                partial_input.add(this.bias);
            }

            const i_hat = Matrix.multiply(this.weights, this.state);
            i_hat.add(this.bias);

            if (partial_input) i_hat.add(partial_input);
            i_hat.transform(this.activation.function);

            this.state = partial_input.clone(); //maybe don't need to clone

            // if (i < this.shape.input[1] - this.shape.output[1] + this.offset) continue;

            this.outputs.push(partial_input);
        }

        this.state.zeros();

        return this.outputs.slice(-this.shape.output[1]);
    }

    // TODO: dynamic input size propagation
    backPropagate(input, output, loss, hyper_parameters) {
        if (!this.outputs.length) throw new Exception();

        this.optimizer.useParameters(hyper_parameters); //make temp

        let b_hat, w_hat, partial_loss;
        for (let i = this.outputs.length - 1; i >= 0; i--) {
            const output = this.outputs[i];
            output.transform(this.activation.derivative);//.flat();

            if (this.outputs.length - i <= this.shape.output[1]) partial_loss = loss[i].flat();

            if (i < this.shape.input[1]) {
                const partial = this.optimizer.step(Matrix.product(output, partial_loss)); //Probably not correct
                partial.scale(1 / this.shape.input[1]);

                if (partial) {
                    b_hat ? b_hat.add(partial) : b_hat = partial;
                    partial.multiply(Matrix.flat(this.inputs[i]).transpose());
                    w_hat ? w_hat.add(partial) : w_hat = partial;
                }
            }

            partial_loss = Matrix.multiply(Matrix.transpose(this.weights), partial_loss);
        }

        if (b_hat && w_hat) {
            this.bias.sub(b_hat);
            this.weights.sub(w_hat);
        }

        return loss;
    }

    static deserialize(data) {
        return super.deserialize(data, new RecurrentLayer());
    }

}