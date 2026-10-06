import Matrix from "../lib/matrix";
import Optimizer, { ParameterUpdate } from "./optimizer";

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

    step(updates: ParameterUpdate[]) {
        for (const { param, gradient } of updates) {
            const step = new Matrix(gradient).scale(this.learningRate);
            if (this.clipping) step.clip(-this.clipping, this.clipping);

            param.sub(step);
        }
    }

    clone(): this {
        return new GradientDescent({
            learningRate: this.learningRate,
            clipping: this.clipping
        }) as this;
    }

}