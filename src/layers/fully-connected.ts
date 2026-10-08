import { Activator, Initialization, Initializer, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type FullyConnectedParams = {
    input: number | [number, number];
    output: number;
    /**
     * @default {@link Sigmoid}
     */
    activation?: Activator;
    /**
     * @default 'xavier'
     */
    initialization?: Initialization;
};

export default class FullyConnectedLayer extends Layer {

    name = 'fcon';
    weights: Matrix;
    bias: Matrix;

    constructor({
        input,
        output,
        activation = new Sigmoid(),
        initialization = 'xavier'
    }: FullyConnectedParams) {
        if (Array.isArray(input)) input = input[0] * input[1];
        super([input, 1], [output, 1], activation, initialization);

        this.weights = this.initializer(output, input);
        this.bias = new Matrix(output, 1);
    }

    propagate(input: Matrix) {
        const output = Matrix.mult(this.weights, input.reshape(...this.input))
            .add(this.bias);

        return this.activation.activate(output);
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) {
        input.reshape(...this.input);
        const gradient = this.activation.derivative(
            output.reshape(...this.output),
            loss.reshape(...this.output)
        );

        loss = Matrix.transpose(this.weights).mult(gradient);

        this.optimizer.tune(this.weights, Matrix.mult(gradient, input.transpose()));
        this.optimizer.tune(this.bias, gradient);
        this.optimizer.flush();

        return loss;
    }

}