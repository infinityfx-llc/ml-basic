import Matrix from "../lib/matrix";
import LoopLayer, { LoopParams } from "./loop";

export default class RecurrentLayer extends LoopLayer<'state' | 'input' | 'output'> {

    name = 'recu';
    weights: Matrix;
    bias: Matrix;

    constructor(args: LoopParams) {
        super(args);

        this.weights = Matrix.random(this.input[0], this.input[0], -1, 1);
        this.bias = new Matrix(this.input[0], 1);
    }

    clear() {
        this.state.set(0);
    }

    forward(input: Matrix, output: boolean) {
        this.store('state', this.state);
        this.store('input', input);

        this.state = Matrix.mult(this.weights, this.state);
        this.state.add(Matrix.mult(this.weights, input));
        this.state.add(this.bias).apply(this.activation.activate);

        if (output) {
            const output = Matrix.mult(this.weights, this.state).add(this.bias).apply(this.activation.activate);
            this.store('output', output);

            return output;
        } else {
            this.store('output', this.state);
        }
    }

    backward(loss: Matrix) {
        const output = this.get('output').apply(this.activation.deactivate),
            gradient = this.optimizer.step(output.scale(loss), false),
            delta = Matrix.mult(gradient, this.get('state').transpose())
                .add(Matrix.mult(gradient, this.get('input').transpose()));

        this.bias.sub(gradient);
        this.weights.sub(delta);

        return Matrix.transpose(this.weights).mult(loss);
    }

}