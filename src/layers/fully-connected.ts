import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type FullyConnectedParams = {
    input: number | [number, number];
    output: number;
    /**
     * @default {@link Sigmoid}
     */
    activation?: Activator;
};

export default class FullyConnectedLayer extends Layer {

    name = 'fcon';
    weights: Matrix;
    bias: Matrix;

    constructor({
        input,
        output,
        activation = new Sigmoid()
    }: FullyConnectedParams) {
        if (Array.isArray(input)) input = input[0] * input[1];
        super([input, 1], [output, 1], activation);

        this.weights = Matrix.random(output, input, -1, 1);
        this.bias = new Matrix(output, 1);
    }

    propagate(input: Matrix) {
        return Matrix.mult(this.weights, input.reshape(...this.input))
            .add(this.bias)
            .apply(this.activation.activate);
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) {
        output.apply(this.activation.deactivate).reshape(...this.output); // check if this reshape is really needed??
        loss.reshape(...this.output);
        input.reshape(...this.input);

        this.optimizer.step(input, output.scale(loss), (input, gradient) => {
            this.bias.sub(gradient);
            this.weights.sub(gradient.mult(input.transpose()));
        });

        return Matrix.transpose(this.weights).mult(loss);
    }

}