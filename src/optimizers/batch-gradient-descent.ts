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

    name = 'BatchGradientDescent';
    batchSize: number;
    private aggregate?: Matrix;

    constructor({
        learningRate = 0.1,
        clipping = 0,
        batchSize = 8
    }: BatchGradientDescentParams = {}) {
        super({ learningRate, clipping });

        this.batchSize = batchSize;
    }

    step(gradient: Matrix, batching = true) {
        this.aggregate ?
            this.aggregate.add(gradient) :
            this.aggregate = new Matrix(gradient);

        if (!batching || this.t % this.batchSize === 0) {
            gradient = super.step(this.aggregate.scale(1 / this.batchSize));
            this.aggregate = undefined;

            return gradient;
        }

        this.t++; // not super elegant
        return new Matrix(gradient).set(0);
    }

}