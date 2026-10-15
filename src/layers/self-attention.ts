import { Initialization, Linear, Softmax } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type SelfAttentionParams = {
    input: number | [number, number];
    output?: number;
    causal?: boolean;
    /**
     * @default 'xavier'
     */
    initialization?: Initialization;
};

export default class SelfAttentionLayer extends Layer {

    name = 'attn';
    causal: boolean;
    scale: number;

    wq: Matrix; // rename
    wk: Matrix;
    wv: Matrix;

    private lastInput: Matrix = new Matrix(0, 0);
    private lastQ: Matrix = new Matrix(0, 0);
    private lastK: Matrix = new Matrix(0, 0);
    private lastV: Matrix = new Matrix(0, 0);
    private lastA: Matrix = new Matrix(0, 0);

    constructor({
        input,
        output,
        causal = false,
        initialization = 'xavier'
    }: SelfAttentionParams) {
        if (!Array.isArray(input)) input = [input, 1];
        if (!output) output = input[1];

        super(input, [input[0], output], new Linear(), initialization);

        this.causal = causal;
        this.scale = 1 / Math.sqrt(output);

        this.wq = this.initializer(input[1], output);
        this.wk = this.initializer(input[1], output);
        this.wv = this.initializer(input[1], output);
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);
        const seqLen = this.input[0];

        this.lastInput = new Matrix(input);

        const Q = this.lastQ = Matrix.mult(input, this.wq);
        const K = this.lastK = Matrix.mult(input, this.wk);
        const V = this.lastV = Matrix.mult(input, this.wv);

        const A = Matrix.multTransB(Q, K).scale(this.scale);

        // Normalize A via row-wise softmax in-place directly on A
        for (let i = 0; i < seqLen; i++) {
            const rowOffset = i * seqLen;

            if (this.causal) {
                for (let j = i + 1; j < seqLen; j++) {
                    A.entries[rowOffset + j] = -1e9;
                }
            }

            // Find row max for numerical stability
            let maxVal = -Infinity;
            for (let j = 0; j < seqLen; j++) {
                if (A.entries[rowOffset + j] > maxVal) maxVal = A.entries[rowOffset + j];
            }

            // Exponentiate & accumulate sum
            let rowSum = 0;
            for (let j = 0; j < seqLen; j++) {
                const expVal = Math.exp(A.entries[rowOffset + j] - maxVal);
                A.entries[rowOffset + j] = expVal;
                rowSum += expVal;
            }

            // Normalize row
            const invSum = 1 / (rowSum || 1);
            for (let j = 0; j < seqLen; j++) {
                A.entries[rowOffset + j] *= invSum;
            }
        }

        this.lastA = A;

        return Matrix.mult(A, V).reshape(...this.output);
    }

    backPropagate(input: Matrix, _: Matrix, loss: Matrix) {
        input.reshape(...this.input);
        loss.reshape(...this.output);

        const seqLen = this.input[0];
        const X = this.lastInput;
        const Q = this.lastQ;
        const K = this.lastK;
        const V = this.lastV;
        const A = this.lastA;

        const dV = Matrix.multTransA(A, loss);
        const dA = Matrix.multTransB(loss, V);

        const dS = new Matrix(seqLen, seqLen);

        for (let i = 0; i < seqLen; i++) {
            const rowOffset = i * seqLen;

            // Compute dot product: sum_k(dA_ik * A_ik)
            let dot = 0;
            const validLimit = this.causal ? i + 1 : seqLen;

            for (let j = 0; j < validLimit; j++) {
                dot += dA.entries[rowOffset + j] * A.entries[rowOffset + j];
            }

            // dS_ij = A_ij * (dA_ij - dot) * scale
            for (let j = 0; j < validLimit; j++) {
                const idx = rowOffset + j;
                dS.entries[idx] = A.entries[idx] * (dA.entries[idx] - dot) * this.scale;
            }
        }

        const dQ = Matrix.mult(dS, K);
        const dK = Matrix.multTransA(dS, Q);

        this.optimizer.tune(this.wq, Matrix.multTransA(X, dQ));
        this.optimizer.tune(this.wk, Matrix.multTransA(X, dK));
        this.optimizer.tune(this.wv, Matrix.multTransA(X, dV));
        this.optimizer.flush();

        const dXq = Matrix.multTransB(dQ, this.wq);
        const dXk = Matrix.multTransB(dK, this.wk);
        const dXv = Matrix.multTransB(dV, this.wv);

        return dXq.add(dXk).add(dXv).reshape(...this.input);
    }

}
