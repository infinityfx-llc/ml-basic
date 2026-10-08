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
    private beta1Hat = 0;
    private beta2Hat = 0;

    constructor({
        learningRate = 0.01,
        learningRateDecay = 0,
        clipping = 0,
        batchSize = 4,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8
    }: AdamParams = {}) {
        super({ learningRate, learningRateDecay, clipping, batchSize });

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    protected process(matrix: Matrix, gradient: Matrix) {
        let m = this.m.get(matrix);
        if (!m) this.m.set(matrix, m = new Matrix(matrix.rows, matrix.columns));

        let v = this.v.get(matrix);
        if (!v) this.v.set(matrix, v = new Matrix(matrix.rows, matrix.columns));

        m.scale(this.beta1).add(new Matrix(gradient).scale(1 - this.beta1));
        v.scale(this.beta2).add(gradient.apply(val => val * val).scale(1 - this.beta2));

        const mHat = new Matrix(m).scale(this.beta1Hat);
        const vHat = new Matrix(v).scale(this.beta2Hat);

        gradient = mHat.scale(vHat.apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val));

        super.process(matrix, gradient);
    }

    flush(force?: boolean) {
        this.t++;
        this.beta1Hat = 1 / (1 - Math.pow(this.beta1, this.t));
        this.beta2Hat = 1 / (1 - Math.pow(this.beta2, this.t));

        super.flush(force);
    }

}