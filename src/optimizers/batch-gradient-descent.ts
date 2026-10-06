import Matrix from "../lib/matrix";
import { ParameterUpdate } from "./optimizer";
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

    step(updates: ParameterUpdate[]) {
        this.i++;

        for (const { param, gradient } of updates) {
            const acc = this.batch.get(param);
            if (acc) {
                acc.add(gradient);
            } else {
                this.batch.set(param, new Matrix(gradient));
            }
        }

        if (this.i % this.batchSize !== 0) return;

        this.applyBatch(this.batchSize);
    }

    flush() {
        const remaining = this.i % this.batchSize;
        if (remaining > 0 && this.batch.size > 0) {
            this.applyBatch(remaining);
        }
    }

    protected applyBatch(count: number) {
        const batchUpdates: ParameterUpdate[] = [];
        for (const [param, acc] of this.batch.entries()) {
            batchUpdates.push({
                param,
                gradient: acc.scale(1 / count)
            });
        }
        this.batch.clear();

        super.step(batchUpdates);
    }

    clone(): this {
        return new BatchGradientDescent({
            learningRate: this.learningRate,
            clipping: this.clipping,
            batchSize: this.batchSize
        }) as this;
    }

}