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
            this.entries = new Float64Array(rowsOrMatrix.entries);

            return;
        }

        this.rows = rowsOrMatrix;
        this.columns = columns || rowsOrMatrix;
        this.entries = new Float64Array(rowsOrMatrix * (columns || rowsOrMatrix));

        if (entries.length) this.entries.set(entries);
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
        if (valueOrMatrix instanceof Matrix) {
            if (!this.isEqualShape(valueOrMatrix)) throw new Error('Additive matrix must have an equal shape');

            for (let i = 0; i < this.entries.length; i++) this.entries[i] += valueOrMatrix.entries[i];
        } else {
            for (let i = 0; i < this.entries.length; i++) this.entries[i] += valueOrMatrix;
        }

        return this;
    }

    sub(n: number): Matrix;
    sub(matrix: Matrix): Matrix;
    sub(valueOrMatrix: number | Matrix) {
        if (valueOrMatrix instanceof Matrix) {
            if (!this.isEqualShape(valueOrMatrix)) throw new Error('Subtractive matrix must have an equal shape');

            for (let i = 0; i < this.entries.length; i++) this.entries[i] -= valueOrMatrix.entries[i];
        } else {
            for (let i = 0; i < this.entries.length; i++) this.entries[i] -= valueOrMatrix;
        }

        return this;
    }

    scale(n: number): Matrix;
    scale(matrix: Matrix): Matrix;
    scale(valueOrMatrix: number | Matrix) {
        if (valueOrMatrix instanceof Matrix) {
            if (!this.isEqualShape(valueOrMatrix)) throw new Error('Scaling matrix must have an equal shape');

            for (let i = 0; i < this.entries.length; i++) this.entries[i] *= valueOrMatrix.entries[i];
        } else {
            for (let i = 0; i < this.entries.length; i++) this.entries[i] *= valueOrMatrix;
        }

        return this;
    }

    apply(func: (value: number) => number) {
        for (let i = 0; i < this.entries.length; i++) this.entries[i] = func(this.entries[i]);

        return this;
    }

    sum() {
        let sum = 0;

        for (let i = 0; i < this.entries.length; i++) sum += this.entries[i];

        return sum;
    }

    mean() {
        return this.entries.length === 0 ? 0 : this.sum() / this.entries.length;
    }

    var(mean = this.mean()) {
        let s = 0;

        for (let i = 0; i < this.entries.length; i++) {
            s += Math.pow(this.entries[i] - mean, 2);
        }

        return s / this.entries.length;
    }

    std(mean?: number) {
        return Math.sqrt(this.var(mean));
    }

    static mult(a: Matrix, b: Matrix) {
        if (a.columns !== b.rows) throw new Error(`Matrix A's columns must be equal to Matrix B's rows`);

        const c = new Matrix(a.rows, b.columns);

        for (let i = 0; i < a.rows; i++) {
            const ra = i * a.columns;
            const rc = i * b.columns;

            for (let k = 0; k < a.columns; k++) {
                const value = a.entries[ra + k];
                if (value === 0) continue;

                const rb = k * b.columns;

                for (let j = 0; j < b.columns; j++) {
                    c.entries[rc + j] += value * b.entries[rb + j];
                }
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
        if (matrix.rows + zeroPadding * 2 < kernel.rows || matrix.columns + zeroPadding * 2 < kernel.columns) throw new Error('Kernel size exceeds Matrix shape size');

        const [rows, cols] = calculatePooledMatrix(matrix.rows, matrix.columns, [kernel.rows, kernel.columns], stride, zeroPadding),
            correlated = new Matrix(rows, cols);

        return correlated.accumulate([kernel.rows, kernel.columns], (sum, mr, mc, kr, kl) => {
            const r = mr * stride - zeroPadding + kr;
            const c = mc * stride - zeroPadding + kl;
            const inBounds = r >= 0 && r < matrix.rows && c >= 0 && c < matrix.columns;
            const value = inBounds ? matrix.entries[r * matrix.columns + c] : 0;

            return sum + value * kernel.entries[kr * kernel.columns + kl];
        });
    }

    correlate(kernel: Matrix, stride = 1, zeroPadding = 0) {
        const correlated = Matrix.correlate(this, kernel, stride, zeroPadding);
        this.entries = correlated.entries;
        this.rows = correlated.rows;
        this.columns = correlated.columns;

        return this;
    }

    static reverseCorrelate(matrix: Matrix, kernel: Matrix, stride = 1, zeroPadding = 0) {
        const correlated = new Matrix(
            Math.floor(matrix.rows + zeroPadding * 2 - (kernel.rows - 1) * stride),
            Math.floor(matrix.columns + zeroPadding * 2 - (kernel.columns - 1) * stride)
        );

        return correlated.accumulate([kernel.rows, kernel.columns], (sum, mr, mc, kr, kl) => {
            const r = kr * stride - zeroPadding + mr;
            const c = kl * stride - zeroPadding + mc;
            const inBounds = r >= 0 && r < matrix.rows && c >= 0 && c < matrix.columns;
            const value = inBounds ? matrix.entries[r * matrix.columns + c] : 0;

            return sum + value * kernel.entries[kr * kernel.columns + kl];
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
        if (matrix.rows + zeroPadding * 2 < window[0] || matrix.columns + zeroPadding * 2 < window[1]) throw new Error('Window size exceeds Matrix shape size');

        const [rows, cols] = calculatePooledMatrix(matrix.rows, matrix.columns, window, stride, zeroPadding),
            pooled = new Matrix(rows, cols);

        return pooled.accumulate(window, (aggregate, mr, mc, kr, kl) => {
            const r = mr * stride - zeroPadding + kr;
            const c = mc * stride - zeroPadding + kl;
            const inBounds = r >= 0 && r < matrix.rows && c >= 0 && c < matrix.columns;
            const value = inBounds ? matrix.entries[r * matrix.columns + c] : 0;

            return pooler(aggregate, value);
        }, initial);
    }

    clip(min: number, max: number) {
        for (let i = 0; i < this.entries.length; i++) this.entries[i] = Math.min(Math.max(this.entries[i], min), max);

        return this;
    }

    dialate(gap: number) {
        if (gap <= 0) return this;

        const rows = this.rows + (this.rows - 1) * gap;
        const columns = this.columns + (this.columns - 1) * gap;
        const entries = new Float64Array(rows * columns);
        gap += 1;

        for (let i = 0; i < this.rows; i++) {
            const ri = i * this.columns;
            const rt = i * gap * columns;

            for (let j = 0; j < this.columns; j++) {
                entries[rt + j * gap] = this.entries[ri + j];
            }
        }

        this.entries = entries;
        this.rows = rows;
        this.columns = columns;

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