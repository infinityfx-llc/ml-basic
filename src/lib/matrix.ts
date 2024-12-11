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
        pooler: (aggregate: number, value: number, index: [number, number]) => number;
    }) {
        if (matrix.rows + zeroPadding < window[0] || matrix.columns + zeroPadding < window[1]) throw new Error('Window size exceeds Matrix shape size');

        const [rows, cols] = calculatePooledMatrix(matrix.rows, matrix.columns, window[0], stride, zeroPadding),
            pooled = new Matrix(rows, cols);

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {

                let aggregate = initial;
                for (let k = 0; k < window[0]; k++) {
                    for (let l = 0; l < window[1]; l++) {
                        const index = (i - zeroPadding * stride + k) * matrix.columns + (j - zeroPadding * stride + l);

                        aggregate = pooler(aggregate, matrix.entries[index] || 0, [k, l]);
                    }
                }

                pooled.entries[i * pooled.columns + j] = aggregate;
            }
        }

        return pooled;
    }

    static convolve(matrix: Matrix, kernel: Matrix, stride = 1, zeroPadding = 0) {
        return this.pool({
            matrix,
            window: [kernel.rows, kernel.columns],
            stride,
            zeroPadding,
            pooler: (sum, val, index) => sum + val * kernel.entries[index[0] * kernel.columns + index[1]]
        });
    }

    convolve(kernel: Matrix, stride = 1, zeroPadding = 0) {
        const convolved = Matrix.convolve(this, kernel, stride, zeroPadding);
        this.entries = convolved.entries;
        this.rows = convolved.rows;
        this.columns = convolved.columns;

        return this;
    }

    clip(min: number, max: number) {
        for (let i = 0; i < this.entries.length; i++) this.entries[i] = Math.min(Math.max(this.entries[i], min), max);

        return this;
    }

    expand(gap: number) {
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

        for (let i = 0; i < n; i++) matrix.entries[n + i] = 1;

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