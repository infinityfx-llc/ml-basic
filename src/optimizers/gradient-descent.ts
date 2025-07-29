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
    clipping?: number;
};

export default class GradientDescent extends Optimizer {

    name = 'sgd';
    learningRate: number;
    clipping: number;

    constructor({
        learningRate = 0.01,
        clipping = 0
    }: GradientDescentParams = {}) {
        super();

        this.learningRate = learningRate;
        this.clipping = clipping;
    }

    step(input: Matrix, gradient: Matrix, callback: (input: Matrix, gradient: Matrix) => void) {
        gradient.scale(this.learningRate);
        if (this.clipping) gradient.clip(-this.clipping, this.clipping);

        callback(input, gradient);
    }

}