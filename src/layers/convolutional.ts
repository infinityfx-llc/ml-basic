import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import { calculatePooledMatrix } from "../lib/utils";
import Layer from "./layer";

export type ConvolutionalParams = {
    input: [number, number];
    kernel: [number, number];
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
        const output = calculatePooledMatrix(...input, kernel, stride, padding);

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
        output.apply(this.activation.deactivate).reshape(...this.output);
        loss.reshape(...this.output);
        input.reshape(...this.input);

        const delta = output.scale(loss);
        loss = new Matrix(delta)
            .dialate(this.stride - 1)
            .correlate(new Matrix(this.kernel).flip(), 1, this.kernel.rows - 1 - this.padding);

        this.optimizer.step(input, delta, (input, gradient) => {
            this.bias.sub(gradient);
            this.kernel.sub(Matrix.reverseCorrelate(input, gradient, this.stride, this.padding));
        });

        return loss;
    }

}