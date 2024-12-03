import Matrix from "../lib/matrix";
import Optimizer from "./optimizer";

export default class GradientDescent extends Optimizer {

    name = 'GradientDescent';
    t = 0;
    learningRate: number;
    clipping: number;

    constructor({
        learningRate = 0.1,
        clipping = 0
    } = {}) {
        super();

        this.learningRate = learningRate;
        this.clipping = clipping;
    }

    step(gradient: Matrix) {
        this.t++;

        gradient.scale(this.learningRate)
        if (this.clipping) gradient.clip(-this.clipping, this.clipping);

        return gradient;
    }

}