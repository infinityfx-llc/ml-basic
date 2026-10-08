import Matrix from "../lib/matrix";
import Optimizer from "./optimizer";

export type GradientDescentParams = {
    /**
     * @default 0.01
     */
    learningRate?: number;
    /**
     * @default 0
     */
    learningRateDecay?: number;
    /**
     * @default 0
     */
    clipping?: number;
};

export default class GradientDescent extends Optimizer {

    name = 'sgd';
    i = 0;
    lr: number;
    learningRate: number;
    learningRateDecay: number;
    clipping: number;

    constructor({
        learningRate = 0.01,
        learningRateDecay = 0,
        clipping = 0
    }: GradientDescentParams = {}) {
        super();

        this.learningRate = this.lr = learningRate;
        this.learningRateDecay = learningRateDecay;
        this.clipping = clipping;
    }

    tune(matrix: Matrix, gradient: Matrix) {
        const step = new Matrix(gradient).scale(this.lr);
        if (this.clipping) step.clip(-this.clipping, this.clipping);

        matrix.sub(step);
    }

    flush() {
        this.i++;

        if (this.learningRateDecay > 0) {
            this.lr = this.learningRate / (1 + this.learningRateDecay * this.i);
        }
    }

}