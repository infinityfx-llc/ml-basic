import { Sigmoid, TanH } from "../lib/functions";
import Matrix from "../lib/matrix";
import LoopLayer, { LoopParams } from "./loop-layer";

export default class LSTMLayer extends LoopLayer<'o' | 'u' | 'c' | 'memory' | 'state'> {

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
            this[`${type}Weights`] = Matrix.random(this.input[0], this.input[0], -1, 1);
            this[`${type}Bias`] = Matrix.random(this.input[0], 1, -1, 1);
        }

        this.memory = new Matrix(this.input[0], 1);
    }

    clear() {
        this.state.set(0);
        this.memory.set(0);
    }

    forward(input: Matrix | undefined, output: boolean) {
        const u = Matrix.mult(this.uWeights, this.state);
        const c = Matrix.mult(this.cWeights, this.state);
        const f = Matrix.mult(this.fWeights, this.state);
        const o = Matrix.mult(this.oWeights, this.state);

        if (input) {
            u.add(Matrix.mult(this.uWeights, input));
            c.add(Matrix.mult(this.cWeights, input));
            f.add(Matrix.mult(this.fWeights, input));
            o.add(Matrix.mult(this.oWeights, input));
        }

        u.add(this.uBias).apply(this.sigmoid.activate);
        this.cache('u', u);

        c.add(this.cBias).apply(this.tanh.activate);
        this.cache('c', c);

        f.add(this.fBias).apply(this.sigmoid.activate);
        o.add(this.oBias).apply(this.sigmoid.activate);
        this.cache('o', o);

        u.scale(c);

        if (!this.index) this.cache('memory', this.memory);
        this.memory.scale(f).add(u);
        this.cache('memory', this.memory);

        this.state = new Matrix(this.memory).apply(this.tanh.activate).scale(o);
        this.cache('state', this.state);

        if (output) return Matrix.mult(this.yWeights, this.state).add(this.yBias).apply(this.activation.activate);
    }

    backward(input: Matrix, output: Matrix, loss: Matrix) {
        const gradient = this.optimizer.step(output.scale(loss), false);

        this.yBias.sub(gradient.scale(1 / this.input[0]));
        this.yWeights.sub(gradient.mult(new Matrix(input).transpose()));

        // from llm, check correctness
        const dState = Matrix.mult(Matrix.transpose(this.yWeights), gradient);
        const dO = this.get('memory', 1).scale(dState).apply(this.sigmoid.deactivate);
        const dMemory = dState.scale(this.get('o'));
        const dF = new Matrix(this.get('memory')).scale(dMemory).apply(this.sigmoid.deactivate);
        const dU = this.get('c').scale(dMemory).apply(this.sigmoid.deactivate);
        const dC = this.get('u').scale(dMemory).apply(this.tanh.deactivate);
        const stateT = this.get('state').transpose();

        this.uBias.sub(dU);
        this.uWeights.sub(dU.mult(stateT));

        this.cBias.sub(dC);
        this.cWeights.sub(dC.mult(stateT));

        this.fBias.sub(dF);
        this.fWeights.sub(dF.mult(stateT));

        this.oBias.sub(dO);
        this.oWeights.sub(dO.mult(stateT));

        return Matrix.transpose(this.uWeights).mult(dU)
            .add(Matrix.transpose(this.cWeights).mult(dC))
            .add(Matrix.transpose(this.fWeights).mult(dF))
            .add(Matrix.transpose(this.oWeights).mult(dO));
    }

}