import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export default class ConvolutionalLayer extends Layer {

    name = 'Convolutional';
    kernel: Matrix;
    stride: number;

    constructor({
        input,
        kernel,
        stride = 1,
        activation = new Sigmoid()
    }: {
        input: [number, number];
        kernel: [number, number];
        stride: number;
        activation?: Activator;
    }) {
        const output: [number, number] = [
            (input[0] - kernel[0]) / stride + 1,
            (input[1] - kernel[1]) / stride + 1
        ];

        super(input, output, activation);
        this.kernel = Matrix.random(kernel[0], kernel[1], -1, 1);
        this.stride = stride;
    }

    propagate(input: Matrix) {
        return Matrix.convolve(input.reshape(...this.input), this.kernel, this.stride).apply(this.activation.activate);
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) {
        output.apply(this.activation.deactivate);
        loss.reshape(...this.output);
        input.reshape(...this.input);
        
        const gradient = this.optimizer.step(output.scale(loss));
        this.kernel.sub(Matrix.convolve(input, gradient).scale(1 / (this.input[0] * this.input[1]))); // scale needed?

        // padding??
        const padding = 0;
        return new Matrix(this.kernel).flip().convolve(loss, this.stride, padding); // stride or 1?
    }

}