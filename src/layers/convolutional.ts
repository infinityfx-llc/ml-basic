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

export default class ConvolutionalLayer extends Layer {

    name = 'Convolutional';
    kernel: Matrix;
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
        this.stride = stride;
        this.padding = padding;
    }

    propagate(input: Matrix) {
        return Matrix.convolve(input.reshape(...this.input), this.kernel, this.stride, this.padding).apply(this.activation.activate);
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) {
        output.apply(this.activation.deactivate).reshape(...this.output); // check if this reshape is really needed??
        loss.reshape(...this.output);
        input.reshape(...this.input);

        const gradient = this.optimizer.step(output.scale(loss));
        this.kernel.sub(Matrix.convolve(input, gradient).scale(1 / (this.input[0] * this.input[1]))); // scale needed?

        return new Matrix(this.kernel).flip().convolve(loss, 1, this.input[0] - this.kernel.rows); // padding only works for symmetry
    }

}