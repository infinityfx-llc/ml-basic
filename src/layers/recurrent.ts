import Matrix from "../lib/matrix";
import LoopLayer, { LoopParams } from "./loop-layer";

export default class RecurrentLayer extends LoopLayer {

    name = 'recu';
    weights: Matrix;
    bias: Matrix;

    constructor(args: LoopParams) {
        super(args);

        this.weights = Matrix.random(this.input[0], this.input[0], -1, 1);
        this.bias = Matrix.random(this.input[0], 1, -1, 1);
    }

    clear() {
        this.state.set(0);
    }

    forward(input: Matrix | undefined, output: boolean) {
        this.state = Matrix.mult(this.weights, this.state);

        if (input) this.state.add(Matrix.mult(this.weights, input));

        this.state.add(this.bias).apply(this.activation.activate);

        if (output) return Matrix.mult(this.weights, this.state).add(this.bias).apply(this.activation.activate);
    }

    backward(input: Matrix, output: Matrix, loss: Matrix) {
        const gradient = this.optimizer.step(output.scale(loss), false);
        this.bias.sub(gradient.scale(1 / this.input[0])); // scale needed?
        this.weights.sub(gradient.mult(new Matrix(input).transpose())); // this input (previous output or step input?) (this needs fixing!!)

        return Matrix.transpose(this.weights).mult(loss);
    }

}