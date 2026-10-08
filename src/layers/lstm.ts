import { Sigmoid, TanH } from "../lib/functions";
import Matrix from "../lib/matrix";
import LoopLayer, { LoopParams } from "./loop";

export default class LSTMLayer extends LoopLayer<'f' | 'o' | 'u' | 'c' | 'memory' | 'state' | 'input' | 'output'> {

    name = 'lstm';
    // @ts-expect-error
    yWeights: Matrix;
    // @ts-expect-error
    yBias: Matrix;
    // @ts-expect-error
    fWeights: Matrix;
    // @ts-expect-error
    fBias: Matrix;
    // @ts-expect-error
    oWeights: Matrix;
    // @ts-expect-error
    oBias: Matrix;
    // @ts-expect-error
    uWeights: Matrix;
    // @ts-expect-error
    uBias: Matrix;
    // @ts-expect-error
    cWeights: Matrix;
    // @ts-expect-error
    cBias: Matrix;
    memory: Matrix;
    sigmoid = new Sigmoid();
    tanh = new TanH();

    constructor(args: LoopParams) {
        super(args);

        for (const type of ['y', 'f', 'o', 'u', 'c'] as const) {
            this[`${type}Weights`] = this.initializer(this.input[0], this.input[0]);
            this[`${type}Bias`] = new Matrix(this.input[0], 1);
        }

        this.memory = new Matrix(this.input[0], 1);
    }

    clear() {
        this.state.set(0);
        this.memory.set(0);
    }

    forward(input: Matrix, output: boolean) {
        this.store('state', this.state);
        this.store('input', input);

        const u = Matrix.mult(this.uWeights, this.state);
        const c = Matrix.mult(this.cWeights, this.state);
        const f = Matrix.mult(this.fWeights, this.state);
        const o = Matrix.mult(this.oWeights, this.state);
        u.add(Matrix.mult(this.uWeights, input));
        c.add(Matrix.mult(this.cWeights, input));
        f.add(Matrix.mult(this.fWeights, input));
        o.add(Matrix.mult(this.oWeights, input));

        this.sigmoid.activate(u.add(this.uBias));
        this.store('u', u);

        this.tanh.activate(c.add(this.cBias));
        this.store('c', c);

        this.sigmoid.activate(f.add(this.fBias));
        this.store('f', f);
        this.sigmoid.activate(o.add(this.oBias));
        this.store('o', o);

        u.scale(c);

        if (!this.index) this.store('memory', this.memory);
        this.memory.scale(f).add(u);
        this.state = this.tanh.activate(new Matrix(this.memory));

        this.store('memory', this.memory);
        this.state.scale(o);

        if (output) {
            const output = this.activation.activate(Matrix.mult(this.yWeights, this.state).add(this.yBias));
            this.store('output', output);

            return output;
        } else {
            this.store('output', this.state);
        }
    }

    backward(loss: Matrix) {
        const gradient = this.activation.derivative(this.get('output'), loss);

        const stateT = this.get('state').transpose(),
            inputT = this.get('input').transpose();

        const dState = Matrix.mult(Matrix.transpose(this.yWeights), gradient);
        const prevMemory = this.get('memory', -1);
        const currentMemory = this.get('memory');
        const tanhMem = this.tanh.activate(new Matrix(currentMemory));
        const oAct = this.get('o');
        const uAct = this.get('u');
        const cAct = this.get('c');
        const fAct = this.get('f');

        const dO = new Matrix(tanhMem).scale(dState).scale(this.sigmoid.deactivate(new Matrix(oAct)));
        const dMemory = this.tanh.deactivate(new Matrix(tanhMem)).scale(dState).scale(oAct);
        const dF = new Matrix(prevMemory).scale(dMemory).scale(this.sigmoid.deactivate(new Matrix(fAct)));
        const dU = new Matrix(cAct).scale(dMemory).scale(this.sigmoid.deactivate(new Matrix(uAct)));
        const dC = new Matrix(uAct).scale(dMemory).scale(this.tanh.deactivate(new Matrix(cAct)));

        const nextLoss = Matrix.transpose(this.uWeights).mult(dU)
            .add(Matrix.transpose(this.cWeights).mult(dC))
            .add(Matrix.transpose(this.fWeights).mult(dF))
            .add(Matrix.transpose(this.oWeights).mult(dO));

        this.optimizer.tune(this.uBias, dU);
        this.optimizer.tune(this.uWeights, Matrix.mult(dU, stateT).add(Matrix.mult(dU, inputT)));
        this.optimizer.tune(this.cBias, dC);
        this.optimizer.tune(this.cWeights, Matrix.mult(dC, stateT).add(Matrix.mult(dC, inputT)));
        this.optimizer.tune(this.fBias, dF);
        this.optimizer.tune(this.fWeights, Matrix.mult(dF, stateT).add(Matrix.mult(dF, inputT)));
        this.optimizer.tune(this.oBias, dO);
        this.optimizer.tune(this.oWeights, Matrix.mult(dO, stateT).add(Matrix.mult(dO, inputT)));
        this.optimizer.tune(this.yBias, gradient);
        this.optimizer.tune(this.yWeights, Matrix.mult(gradient, stateT).add(Matrix.mult(gradient, inputT)));
        this.optimizer.flush();

        return nextLoss;
    }

}