import Matrix from "../lib/matrix";
import LoopLayer, { LoopParams } from "./loop";

export default class RecurrentLayer extends LoopLayer<'state' | 'input' | 'output'> {

    name = 'recu';
    iWeights: Matrix;
    hWeights: Matrix;
    bias: Matrix;

    constructor(args: LoopParams) {
        super(args);

        this.iWeights = this.initializer(this.input[0], this.input[0]);
        this.hWeights = this.initializer(this.input[0], this.input[0]);
        this.bias = new Matrix(this.input[0], 1);
    }

    clear() {
        this.state.set(0);
    }

    forward(input: Matrix, output: boolean) {
        this.store('state', this.state);
        this.store('input', input);

        this.state = Matrix.mult(this.hWeights, this.state);
        this.state.add(Matrix.mult(this.iWeights, input));
        this.activation.activate(this.state.add(this.bias));

        if (output) {
            this.store('output', this.state);

            return this.state;
        } else {
            this.store('output', this.state);
        }
    }

    backward(loss: Matrix) {
        const gradient = this.activation.derivative(this.get('output'), loss);

        const dInputWeights = Matrix.mult(gradient, this.get('input').transpose());
        const dRecurrentWeights = Matrix.mult(gradient, this.get('state').transpose());

        this.optimizer.tune(this.bias, gradient);
        this.optimizer.tune(this.iWeights, dInputWeights);
        this.optimizer.tune(this.hWeights, dRecurrentWeights);
        this.optimizer.flush();

        return Matrix.transpose(this.hWeights).mult(gradient);
    }

}