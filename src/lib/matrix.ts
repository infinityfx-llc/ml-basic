import { calculatePooledMatrix } from "./utils";

export default class Matrix {

    rows: number;
    columns: number;
    entries: Float64Array;

    constructor(matrix: Matrix);
    constructor(rows: number, columns: number, entries?: number[] | Float64Array);
    constructor(rowsOrMatrix: number | Matrix, columns?: number, entries = []) {
        if (rowsOrMatrix instanceof Matrix) {
            this.rows = rowsOrMatrix.rows;
            this.columns = rowsOrMatrix.columns;
            this.entries = new Float64Array(rowsOrMatrix.entries.slice());

            return;
        }

        this.rows = rowsOrMatrix;
        this.columns = columns || rowsOrMatrix;
        this.entries = new Float64Array(rowsOrMatrix * (columns || rowsOrMatrix));

        for (let i = 0; i < this.entries.length; i++) this.entries[i] = entries[i] || 0;
    }

    isEqualShape(matrix: Matrix) {
        return this.rows === matrix.rows && this.columns === matrix.columns;
    }

    reshape(rows: number, columns: number) {
        if (rows * columns !== this.entries.length) throw new Error('New shape size must be equal to previous size');

        this.rows = rows;
        this.columns = columns;

        return this;
    }

    flat() {
        return this.reshape(this.entries.length, 1);
    }

    flip() {
        this.entries.reverse();

        return this;
    }

    set(value: number) {
        this.entries.fill(value);

        return this;
    }

    add(n: number): Matrix;
    add(matrix: Matrix): Matrix;
    add(valueOrMatrix: number | Matrix) {
        if (valueOrMatrix instanceof Matrix && !this.isEqualShape(valueOrMatrix)) throw new Error('Additive matrix must have an equal shape');

        for (let i = 0; i < this.entries.length; i++) this.entries[i] += valueOrMatrix instanceof Matrix ? valueOrMatrix.entries[i] : valueOrMatrix;

        return this;
    }

    sub(n: number): Matrix;
    sub(matrix: Matrix): Matrix;
    sub(valueOrMatrix: number | Matrix) {
        if (valueOrMatrix instanceof Matrix && !this.isEqualShape(valueOrMatrix)) throw new Error('Subtractive matrix must have an equal shape');

        for (let i = 0; i < this.entries.length; i++) this.entries[i] -= valueOrMatrix instanceof Matrix ? valueOrMatrix.entries[i] : valueOrMatrix;

        return this;
    }

    scale(n: number): Matrix;
    scale(matrix: Matrix): Matrix;
    scale(valueOrMatrix: number | Matrix) {
        if (valueOrMatrix instanceof Matrix && !this.isEqualShape(valueOrMatrix)) throw new Error('Scaling matrix must have an equal shape');

        for (let i = 0; i < this.entries.length; i++) this.entries[i] *= valueOrMatrix instanceof Matrix ? valueOrMatrix.entries[i] : valueOrMatrix;

        return this;
    }

    apply(func: (value: number) => number) {
        for (let i = 0; i < this.entries.length; i++) this.entries[i] = func(this.entries[i]);

        return this;
    }

    sum() {
        return this.entries.reduce((a, b) => a + b, 0);
    }

    static mult(a: Matrix, b: Matrix) {
        if (a.columns !== b.rows) throw new Error(`Matrix A's columns must be equal to Matrix B's rows`);

        const c = new Matrix(a.rows, b.columns);

        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.columns; j++) {
                let sum = 0;

                for (let k = 0; k < a.columns; k++) {
                    sum += a.entries[i * a.columns + k] * b.entries[k * b.columns + j];
                }

                c.entries[i * b.columns + j] = sum;
            }
        }

        return c;
    }

    mult(matrix: Matrix) {
        matrix = Matrix.mult(this, matrix);

        this.entries = matrix.entries;
        this.columns = matrix.columns;

        return this;
    }

    static transpose(matrix: Matrix) {
        const transposed = new Matrix(matrix.columns, matrix.rows);

        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.columns; j++) {
                transposed.entries[j * matrix.rows + i] = matrix.entries[i * matrix.columns + j];
            }
        }

        return transposed;
    }

    transpose() {
        const transposed = Matrix.transpose(this);

        this.entries = transposed.entries;
        this.rows = transposed.rows;
        this.columns = transposed.columns;

        return this;
    }

    private accumulate(
        window: [number, number],
        accumulator: (aggregate: number, mr: number, mc: number, kr: number, kl: number) => number,
        initial = 0
    ) {

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.columns; j++) {
                let aggregate = initial;

                for (let k = 0; k < window[0]; k++) {
                    for (let l = 0; l < window[1]; l++) {
                        aggregate = accumulator(aggregate, i, j, k, l);
                    }
                }

                this.entries[i * this.columns + j] = aggregate;
            }
        }

        return this;
    }

    static correlate(matrix: Matrix, kernel: Matrix, stride = 1, zeroPadding = 0) {
        if (matrix.rows + zeroPadding < kernel.rows || matrix.columns + zeroPadding < kernel.columns) throw new Error('Kernel size exceeds Matrix shape size');

        const [rows, cols] = calculatePooledMatrix(matrix.rows, matrix.columns, kernel.rows, stride, zeroPadding),
            correlated = new Matrix(rows, cols);

        return correlated.accumulate([kernel.rows, kernel.columns], (sum, mr, mc, kr, kl) => {
            const value = matrix.entries[(mr - zeroPadding * stride + kr) * matrix.columns + (mc - zeroPadding * stride + kl)];

            return sum + (value || 0) * kernel.entries[kr * kernel.columns + kl];
        });
    }

    correlate(kernel: Matrix, stride = 1, zeroPadding = 0) {
        const correlated = Matrix.correlate(this, kernel, stride, zeroPadding);
        this.entries = correlated.entries;
        this.rows = correlated.rows;
        this.columns = correlated.columns;

        return this;
    }

    static reverseCorrelate(matrix: Matrix, kernel: Matrix, stride = 1) {
        const correlated = new Matrix(
            Math.floor(matrix.rows - (kernel.rows - 1) * stride),
            Math.floor(matrix.columns - (kernel.columns - 1) * stride)
        );

        return correlated.accumulate([kernel.rows, kernel.columns], (sum, mr, mc, kr, kl) => {
            const value = matrix.entries[(mr + kr * stride) * matrix.columns + (mc + kl * stride)];

            return sum + (value || 0) * kernel.entries[kr * kernel.columns + kl];
        });
    }

    static pool({
        matrix,
        window,
        stride = 1,
        zeroPadding = 0,
        initial = 0,
        pooler
    }: {
        matrix: Matrix;
        window: [number, number];
        stride?: number;
        zeroPadding?: number;
        initial?: number;
        pooler: (aggregate: number, value: number) => number;
    }) {
        if (matrix.rows + zeroPadding < window[0] || matrix.columns + zeroPadding < window[1]) throw new Error('Window size exceeds Matrix shape size');

        const [rows, cols] = calculatePooledMatrix(matrix.rows, matrix.columns, window[0], stride, zeroPadding),
            pooled = new Matrix(rows, cols);

        return pooled.accumulate(window, (aggregate, mr, mc, kr, kl) => {
            const value = matrix.entries[(mr - zeroPadding * stride + kr) * matrix.columns + (mc - zeroPadding * stride + kl)];

            return pooler(aggregate, value || 0);
        }, initial);
    }

    clip(min: number, max: number) {
        for (let i = 0; i < this.entries.length; i++) this.entries[i] = Math.min(Math.max(this.entries[i], min), max);

        return this;
    }

    dialate(gap: number) {
        const columns = this.columns,
            entries = this.entries;

        this.rows += (this.rows - 1) * gap;
        this.columns += (this.columns - 1) * gap;
        this.entries = new Float64Array(this.rows * this.columns);
        gap += 1;

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.columns; j++) {
                this.entries[i * this.columns + j] = i % gap == 0 && j % gap == 0 ?
                    entries[Math.floor(i / gap) * columns + Math.floor(j / gap)] :
                    0;
            }
        }

        return this;
    }

    static identity(n: number) {
        const matrix = new Matrix(n, n);

        for (let i = 0; i < n; i++) matrix.entries[i * n + i] = 1;

        return matrix;
    }

    static random(rows: number, columns: number, min = 0, max = 1) {
        const matrix = new Matrix(rows, columns);

        for (let i = 0; i < matrix.entries.length; i++) matrix.entries[i] = Math.random() * (max - min) + min;

        return matrix;
    }

    serialize() {
        return {
            type: 'Matrix',
            rows: this.rows,
            columns: this.columns,
            entries: Array.from(this.entries)
        };
    }

}