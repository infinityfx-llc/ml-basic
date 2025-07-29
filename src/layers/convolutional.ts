import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import { calculatePooledMatrix } from "../lib/utils";
import Layer from "./layer";

export type ConvolutionalParams = {
    input: [number, number];
    kernel: [number, number]; // just allow for one value?? (symmetric)
    /**
     * @default 1
     */
    stride?: number;
    /**
     * @default 0
     */
    padding?: number;
    /**
     * @default {@link Sigmoid}
     */
    activation?: Activator;
};

// non square support

export default class ConvolutionalLayer extends Layer {

    name = 'conv';
    kernel: Matrix;
    bias: Matrix;
    stride: number;
    padding: number;

    constructor({
        input,
        kernel,
        stride = 1,
        padding = 0,
        activation = new Sigmoid()
    }: ConvolutionalParams) {
        const output = calculatePooledMatrix(...input, kernel[0], stride, padding);

        super(input, output, activation);
        this.kernel = Matrix.random(kernel[0], kernel[1], -1, 1);
        this.bias = new Matrix(output[0], output[1]);
        this.stride = stride;
        this.padding = padding;
    }

    propagate(input: Matrix) {
        return Matrix.correlate(input.reshape(...this.input), this.kernel, this.stride, this.padding)
            .add(this.bias)
            .apply(this.activation.activate);
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) {
        output.apply(this.activation.deactivate).reshape(...this.output); // check if this reshape is really needed??
        loss.reshape(...this.output);
        input.reshape(...this.input);

        this.optimizer.step(input, output.scale(loss), (input, gradient) => {
            this.bias.sub(gradient);
            this.kernel.sub(Matrix.reverseCorrelate(input, gradient, this.stride));
        });

        return new Matrix(this.kernel).flip().correlate(loss.dialate(this.stride - 1), 1, this.input[0] - this.kernel.rows); // padding only works for symmetry (also stride)
    }

}