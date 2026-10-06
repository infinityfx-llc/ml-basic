import Matrix from "../lib/matrix";
import BatchGradientDescent from "./batch-gradient-descent";

export type RMSPropParams = {
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
     * @default 1e-8
     */
    epsilon?: number;
};

export default class RMSProp extends BatchGradientDescent {

    name = 'rmsp';
    beta1: number;
    epsilon: number;
    private v = new Map<Matrix, Matrix>();

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 4,
        beta1 = 0.9,
        epsilon = 1e-8
    }: RMSPropParams = {}) {
        super({ learningRate, clipping, batchSize });

        this.beta1 = beta1;
        this.epsilon = epsilon;
    }

    // check for optimization and renaming
    protected applyBatch(count: number) {
        for (const [param, acc] of this.batch.entries()) {
            const g = acc.scale(1 / count);

            let vParam = this.v.get(param);
            if (!vParam) {
                vParam = new Matrix(param.rows, param.columns);
                this.v.set(param, vParam);
            }

            vParam.scale(this.beta1).add(new Matrix(g).apply(val => val * val).scale(1 - this.beta1));

            const step = new Matrix(g)
                .scale(new Matrix(vParam).apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val))
                .scale(this.learningRate);

            if (this.clipping) step.clip(-this.clipping, this.clipping);

            param.sub(step);
        }

        this.batch.clear();
    }

    clone(): this {
        return new RMSProp({
            learningRate: this.learningRate,
            clipping: this.clipping,
            batchSize: this.batchSize,
            beta1: this.beta1,
            epsilon: this.epsilon
        }) as this;
    }

}