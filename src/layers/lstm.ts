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

        u.add(this.uBias).apply(this.sigmoid.activate);
        this.store('u', u);

        c.add(this.cBias).apply(this.tanh.activate);
        this.store('c', c);

        f.add(this.fBias).apply(this.sigmoid.activate);
        this.store('f', f);
        o.add(this.oBias).apply(this.sigmoid.activate);
        this.store('o', o);

        u.scale(c);

        if (!this.index) this.store('memory', this.memory);
        this.memory.scale(f).add(u);
        this.state = new Matrix(this.memory).apply(this.tanh.activate);

        this.store('memory', this.memory);
        this.state.scale(o);

        if (output) {
            const output = Matrix.mult(this.yWeights, this.state).add(this.yBias).apply(this.activation.activate);
            this.store('output', output);

            return output;
        } else {
            this.store('output', this.state);
        }
    }

    backward(loss: Matrix) {
        let output = this.get('output').apply(this.activation.deactivate);
        const gradient = output.scale(loss);

        const stateT = this.get('state').transpose(),
            inputT = this.get('input').transpose();

        const dState = Matrix.mult(Matrix.transpose(this.yWeights), gradient);
        const prevMemory = this.get('memory', -1);
        const currentMemory = this.get('memory');
        const tanhMem = new Matrix(currentMemory).apply(this.tanh.activate);
        const oAct = this.get('o');
        const uAct = this.get('u');
        const cAct = this.get('c');
        const fAct = this.get('f');

        const dO = new Matrix(tanhMem).scale(dState).scale(new Matrix(oAct).apply(this.sigmoid.deactivate));
        const dMemory = new Matrix(tanhMem).apply(this.tanh.deactivate).scale(dState).scale(oAct);
        const dF = new Matrix(prevMemory).scale(dMemory).scale(new Matrix(fAct).apply(this.sigmoid.deactivate));
        const dU = new Matrix(cAct).scale(dMemory).scale(new Matrix(uAct).apply(this.sigmoid.deactivate));
        const dC = new Matrix(uAct).scale(dMemory).scale(new Matrix(cAct).apply(this.tanh.deactivate));

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