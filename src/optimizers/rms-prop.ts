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
    learningRateDecay?: number;
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
        learningRateDecay = 0,
        clipping = 0,
        batchSize = 4,
        beta1 = 0.9,
        epsilon = 1e-8
    }: RMSPropParams = {}) {
        super({ learningRate, learningRateDecay, clipping, batchSize });

        this.beta1 = beta1;
        this.epsilon = epsilon;
    }

    protected process(matrix: Matrix, gradient: Matrix) {
        let v = this.v.get(matrix);
        if (!v) this.v.set(matrix, v = new Matrix(matrix.rows, matrix.columns));

        v.scale(this.beta1).add(new Matrix(gradient).apply(val => val * val).scale(1 - this.beta1));

        gradient.scale(new Matrix(v).apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val));

        super.process(matrix, gradient);
    }

}