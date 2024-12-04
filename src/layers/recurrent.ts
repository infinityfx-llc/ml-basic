import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export default class RecurrentLayer extends Layer {

    name = 'Recurrent';
    weights: Matrix;
    bias: Matrix;
    state: Matrix;

    constructor({
        input,
        output,
        activation = new Sigmoid()
    }: {
        input: [number, number];
        output: number;
        // offset?
        activation?: Activator;
    }) {
        super(input, [input[0], output], activation);

        this.weights = Matrix.random(input[0], input[0], -1, 1);
        this.bias = Matrix.random(input[0], 1, -1, 1);
        this.state = new Matrix(input[0], 1);
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        const output = [],
            len = Math.max(this.input[1], this.output[1]);

        for (let i = 0; i < len; i++) {

            this.state = Matrix.mult(this.weights, this.state);

            if (i < this.input[1] - 1) {
                const stepInput = new Matrix(this.input[0], 1,
                    input.entries.slice(i * this.input[0], (i + 1) * this.input[0]) as any as number[]);
                this.state.add(Matrix.mult(this.weights, stepInput));
            }

            this.state.add(this.bias).apply(this.activation.activate);

            if (i >= len - this.output[1]) {
                const stepOutput = Matrix.mult(this.weights, this.state).add(this.bias).apply(this.activation.activate);
                output.push(...stepOutput.entries);
            }
        }

        this.state.set(0);

        return new Matrix(...this.output, output);
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) {

        // todo
        return new Matrix(...this.input);
    }

}