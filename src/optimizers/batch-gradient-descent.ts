import Matrix from "../lib/matrix";
import GradientDescent from "./gradient-descent";

export type BatchGradientDescentParams = {
    /**
     * @default 0.01
     */
    learningRate?: number;
    /**
     * @default 0
     */
    clipping?: number;
    /**
     * @default 8
     */
    batchSize?: number;
};

export default class BatchGradientDescent extends GradientDescent {

    name = 'bgd';
    i = 0;
    batchSize: number;
    private batch?: {
        input: Matrix;
        gradient: Matrix;
    };

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 8
    }: BatchGradientDescentParams = {}) {
        super({ learningRate, clipping });

        this.batchSize = batchSize;
    }

    step(input: Matrix, gradient: Matrix, callback: (input: Matrix, gradient: Matrix) => void) {
        if (this.batch) {
            this.batch.input.add(input);
            this.batch.gradient.add(gradient);
        } else {
            this.batch = {
                input: new Matrix(input),
                gradient: new Matrix(gradient)
            };
        }

        this.i++;

        if (this.i % this.batchSize !== 0 || !this.batch) return;

        super.step(this.batch.input.scale(1 / this.batchSize), this.batch.gradient.scale(1 / this.batchSize), callback);

        this.batch = undefined;
    }

}