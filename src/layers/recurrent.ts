import Matrix from "../lib/matrix";
import LoopLayer, { LoopParams } from "./loop";

export default class RecurrentLayer extends LoopLayer<'state' | 'input' | 'output'> {

    name = 'recu';
    inputWeights: Matrix;
    recurrentWeights: Matrix;
    bias: Matrix;

    constructor(args: LoopParams) {
        super(args);

        this.inputWeights = Matrix.random(this.input[0], this.input[0], -1, 1);
        this.recurrentWeights = Matrix.random(this.input[0], this.input[0], -1, 1);
        this.bias = new Matrix(this.input[0], 1);
    }

    clear() {
        this.state.set(0);
    }

    forward(input: Matrix, output: boolean) {
        this.store('state', this.state);
        this.store('input', input);

        this.state = Matrix.mult(this.recurrentWeights, this.state);
        this.state.add(Matrix.mult(this.inputWeights, input));
        this.state.add(this.bias).apply(this.activation.activate);

        if (output) {
            this.store('output', this.state);

            return this.state;
        } else {
            this.store('output', this.state);
        }
    }

    backward(loss: Matrix) {
        const output = this.get('output').apply(this.activation.deactivate);
        const gradient = output.scale(loss);

        const dInputWeights = Matrix.mult(gradient, this.get('input').transpose());
        const dRecurrentWeights = Matrix.mult(gradient, this.get('state').transpose());

        this.optimizer.step([
            { param: this.bias, gradient },
            { param: this.inputWeights, gradient: dInputWeights },
            { param: this.recurrentWeights, gradient: dRecurrentWeights }
        ]);

        return Matrix.transpose(this.recurrentWeights).mult(gradient);
    }

}