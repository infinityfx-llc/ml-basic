const Sigmoid = require('../functions/sigmoid');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Layer = require('./layer');

module.exports = class ConvolutionalLayer extends Layer {

    static name = 'convolutional';

    constructor({ input = [8, 8], kernel = [5, 5], stride = 1, activation = Sigmoid, optimizer = BatchGradientDescent, hyper_parameters = {} } = {}) {
        super(activation, optimizer, hyper_parameters);

        this.shape = {
            input,
            output: [
                Math.ceil((input[0] - kernel[0]) / stride + 1),
                Math.ceil((input[1] - kernel[1]) / stride + 1)
            ]
        };
        this.stride = stride;
        this.kernel = Matrix.random(kernel[0], kernel[1], -1, 1);
    }

    flush() {
        this.kernel = Matrix.random(this.kernel.rows, this.kernel.columns, -1, 1);
        super.flush();
    }

    shapeInput(input) {
        return this.shapeData(input, this.shape.input);
    }

    propagate(input) {
        input = Matrix.convolve(this.shapeInput(input), this.kernel, this.stride);

        return input.transform(this.activation.function);
    }

    backPropagate(input, output, loss, hyper_parameters) {
        output.transform(this.activation.derivative)
            .reshape(...this.shape.output);

        input = Matrix.reshape(input, ...this.shape.input);
        loss.reshape(...this.shape.output);

        this.optimizer.useParameters(hyper_parameters); //make temp
        const gradient = this.optimizer.step(Matrix.product(output, loss));

        if (gradient) {
            this.kernel.add(Matrix.convolve(input, gradient));
        }

        //TODO: check if stride has effect on these operations
        //TODO: implement horizontal and vertical padding
        const padding = this.shape.input[0] - this.kernel.rows;
        //return Matrix.transpose(this.kernel).convolve(loss, 1, padding);
        return Matrix.flip(this.kernel).convolve(loss, 1, padding); //maybe use 180deg rotated kernel instead of transpose
    }

    cross(b) {
        this.kernel = Matrix.add(this.kernel, b.kernel).scale(0.5);

        return this;
    }

    mutate({ mutation_probability = 0.2, mutation_constant = 0.5 } = {}) {
        const k_hat = Matrix.random(this.kernel.rows, this.kernel.columns, -mutation_constant, mutation_constant)
            .transform(val => Math.pow(val, Math.round(Math.sqrt(1 / mutation_probability))));
        this.kernel.add(k_hat);
    }

    static deserialize(data) {
        return super.deserialize(data, new ConvolutionalLayer());
    }

}