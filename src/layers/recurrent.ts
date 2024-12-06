import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type RecurrentParams = {
    input: [number, number];
    output: number;
    // offset?
    /**
     * @default {@link Sigmoid}
     */
    activation?: Activator;
};

export default class RecurrentLayer extends Layer {

    name = 'recu';
    weights: Matrix;
    bias: Matrix;
    state: Matrix;
    private outputs: Matrix[];

    constructor({
        input,
        output,
        activation = new Sigmoid()
    }: RecurrentParams) {
        super(input, [input[0], output], activation);

        this.weights = Matrix.random(input[0], input[0], -1, 1);
        this.bias = Matrix.random(input[0], 1, -1, 1);
        this.state = new Matrix(input[0], 1);
        this.outputs = new Array(Math.max(this.input[1], this.output[1]) + 1);
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

                if (i === 0) this.outputs[0] = stepInput;
            }

            this.state.add(this.bias).apply(this.activation.activate);

            if (i >= len - this.output[1]) {
                const stepOutput = Matrix.mult(this.weights, this.state).add(this.bias).apply(this.activation.activate);
                output.push(...stepOutput.entries);

                this.outputs[i + 1] = stepOutput;
            } else {
                this.outputs[i + 1] = new Matrix(this.state);
            }
        }

        this.state.set(0);

        return new Matrix(...this.output, output);
    }

    backPropagate(_1: Matrix, _2: Matrix, loss: Matrix) {
        const len = Math.max(this.input[1], this.output[1]),
            inputLoss = [];

        loss = new Matrix(this.output[0], 1,
            loss.entries.slice((this.output[1] - 1) * this.output[0]) as any as number[]);

        for (let i = len - 1; i >= 0; i--) {
            this.outputs[i + 1].apply(this.activation.deactivate);

            const gradient = this.optimizer.step(this.outputs[i + 1].scale(loss), false);
            this.bias.sub(gradient.scale(1 / this.input[0])); // scale needed?
            this.weights.sub(gradient.mult(new Matrix(this.outputs[i]).transpose())); // this input (previous output or step input?)

            loss = Matrix.transpose(this.weights).mult(loss);

            if (i < this.input[1]) {
                inputLoss.push(...loss.entries);
            }
        }

        return new Matrix(...this.input, inputLoss);
    }

}