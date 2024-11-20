import Matrix from "../lib/matrix";
import GradientDescent from "./gradient-descent";

export default class RMSProp extends GradientDescent {

    batchSize: number;
    beta: number;
    epsilon: number;
    v?: Matrix;

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 4,
        beta = 0.9,
        epsilon = 1e-8
    } = {}) {
        super({ learningRate, clipping });

        this.batchSize = batchSize;
        this.beta = beta;
        this.epsilon = epsilon;
    }

    step(gradient: Matrix) {
        if (!this.v) this.v = new Matrix(gradient).set(1);

        this.v.scale(this.beta).add(new Matrix(gradient).apply(val => val * val).scale(1 - this.beta));

        const partial = this.t % this.batchSize !== 0;

        if (!partial) gradient.scale(new Matrix(this.v).apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val));

        return super.step(partial ? new Matrix(gradient).set(0) : gradient);
    }

}