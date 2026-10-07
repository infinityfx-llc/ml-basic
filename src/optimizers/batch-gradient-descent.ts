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
    protected batch = new Map<Matrix, Matrix>();

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 8
    }: BatchGradientDescentParams = {}) {
        super({ learningRate, clipping });

        this.batchSize = batchSize;
    }

    tune(matrix: Matrix, gradient: Matrix) {
        const entry = this.batch.get(matrix);

        if (entry) {
            entry.add(gradient);
        } else {
            this.batch.set(matrix, new Matrix(gradient));
        }
    }

    protected process(matrix: Matrix, gradient: Matrix) {
        super.tune(matrix, gradient);
    }

    flush(force = false) {
        this.i++;

        const remainder = this.i % this.batchSize;
        if (!force && remainder !== 0) return;
        if (this.batch.size === 0) return;

        const scale = 1 / (remainder === 0 ? this.batchSize : remainder);

        for (const [matrix, gradient] of this.batch.entries()) {
            this.process(matrix, gradient.scale(scale));
        }

        this.batch.clear();
    }

}