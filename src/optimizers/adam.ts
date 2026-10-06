import Matrix from "../lib/matrix";
import BatchGradientDescent from "./batch-gradient-descent";

export type AdamParams = {
    /**
     * @default 0.01
     */
    learningRate?: number;
    /**
     * @default 0
     */
    clipping?: number;
    /**
     * @default 4
     */
    batchSize?: number;
    /**
     * @default 0.9
     */
    beta1?: number;
    /**
     * @default 0.999
     */
    beta2?: number;
    /**
     * @default 1e-8
     */
    epsilon?: number;
};

export default class Adam extends BatchGradientDescent {

    name = 'adam';
    beta1: number;
    beta2: number;
    epsilon: number;
    t = 0;
    private m = new Map<Matrix, Matrix>();
    private v = new Map<Matrix, Matrix>();

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 4,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8
    }: AdamParams = {}) {
        super({ learningRate, clipping, batchSize });

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    protected applyBatch(count: number) {
        this.t++;
        const beta1Correction = 1 / (1 - Math.pow(this.beta1, this.t));
        const beta2Correction = 1 / (1 - Math.pow(this.beta2, this.t));

        for (const [param, acc] of this.batch.entries()) {
            const g = acc.scale(1 / count);

            let mParam = this.m.get(param);
            if (!mParam) {
                mParam = new Matrix(param.rows, param.columns);
                this.m.set(param, mParam);
            }

            let vParam = this.v.get(param);
            if (!vParam) {
                vParam = new Matrix(param.rows, param.columns);
                this.v.set(param, vParam);
            }

            mParam.scale(this.beta1).add(new Matrix(g).scale(1 - this.beta1));
            vParam.scale(this.beta2).add(new Matrix(g).apply(val => val * val).scale(1 - this.beta2));

            const mHat = new Matrix(mParam).scale(beta1Correction);
            const vHat = new Matrix(vParam).scale(beta2Correction);

            const step = mHat
                .scale(vHat.apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val))
                .scale(this.learningRate);

            if (this.clipping) step.clip(-this.clipping, this.clipping);

            param.sub(step);
        }

        this.batch.clear();
    }

    clone(): this {
        return new Adam({
            learningRate: this.learningRate,
            clipping: this.clipping,
            batchSize: this.batchSize,
            beta1: this.beta1,
            beta2: this.beta2,
            epsilon: this.epsilon
        }) as this;
    }

}