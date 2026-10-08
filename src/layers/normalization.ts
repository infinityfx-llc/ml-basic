import { Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type NormalizationParams = {
    input: number | [number, number];
    /**
     * @default 1e-5
     */
    epsilon?: number;
};

export default class NormalizationLayer extends Layer {

    name = 'norm';
    gamma: Matrix;
    beta: Matrix;
    epsilon: number;
    private mean = 0;
    private variance = 0;
    private normalized?: Matrix;

    constructor({
        input,
        epsilon = 1e-5
    }: NormalizationParams) {
        if (!Array.isArray(input)) input = [input, 1];
        super(input, input, new Sigmoid());

        this.epsilon = epsilon;
        this.gamma = new Matrix(input[0], input[1]).set(1);
        this.beta = new Matrix(input[0], input[1]).set(0);
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        this.mean = input.mean();
        this.variance = input.var(this.mean);

        this.normalized = new Matrix(input).sub(this.mean).scale(1 / Math.sqrt(this.variance + this.epsilon));

        return new Matrix(this.normalized)
            .scale(this.gamma)
            .add(this.beta);
    }

    backPropagate(_1: Matrix, _2: Matrix, loss: Matrix) {
        loss.reshape(...this.output);

        const xHat = this.normalized ?? new Matrix(this.input[0], this.input[1]);

        this.optimizer.tune(this.beta, loss);
        const dg = new Matrix(loss).scale(xHat);

        loss.scale(this.gamma);
        xHat.scale(new Matrix(loss).scale(xHat).mean());

        this.optimizer.tune(this.gamma, dg);
        this.optimizer.flush();

        return loss.sub(loss.mean())
            .sub(xHat)
            .scale(1 / Math.sqrt(this.variance + this.epsilon));
    }

}
