import Matrix from "../lib/matrix";
import GradientDescent from "./gradient-descent";

export default class Adam extends GradientDescent {

    batchSize: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    m?: Matrix;
    v?: Matrix;

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 4,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8
    } = {}) {
        super({ learningRate, clipping });

        this.batchSize = batchSize;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    step(gradient: Matrix) {
        if (!this.m || !this.v) {
            this.m = new Matrix(gradient.rows, gradient.columns);
            this.v = new Matrix(this.m);
        }

        this.m.scale(this.beta1).add(new Matrix(gradient).scale(1 - this.beta1));
        this.v.scale(this.beta2).add(new Matrix(gradient).apply(val => val * val).scale(1 - this.beta2));

        const partial = this.t % this.batchSize !== 0;

        if (!partial) {
            const exp = Math.floor(this.t / this.batchSize) + 1;
            const mHat = new Matrix(this.m).scale(1 / (1 - Math.pow(this.beta1, exp))),
                vHat = new Matrix(this.v).scale(1 / (1 - Math.pow(this.beta2, exp)));

            gradient = mHat.scale(vHat.apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val));
        }

        return super.step(partial ? new Matrix(gradient).set(0) : gradient);
    }

}