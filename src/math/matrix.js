const IllegalArgumentException = require('../exceptions/illegal-argument');

module.exports = class Matrix {

    constructor(rows, columns = rows, entries = []) {
        if (rows instanceof Matrix) return Matrix.copy(rows);
        if (rows < 1 || columns < 1) throw new IllegalArgumentException('Rows and Columns must be numbers greater than 0');

        this.rows = rows;
        this.columns = columns;

        if (entries instanceof Float64Array) return (this.entries = entries, this);

        const bytes = Float64Array.BYTES_PER_ELEMENT * rows * columns;
        this.buffer = typeof SharedArrayBuffer !== 'undefined' ? new SharedArrayBuffer(bytes) : new ArrayBuffer(bytes);
        this.entries = new Float64Array(this.buffer);
        for (let i = 0; i < this.entries.length; i++) this.entries[i] = entries[i] || 0;
    }

    static copy(matrix) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix.rows, matrix.columns, matrix.entries.slice());
    }

    isEqualShape(matrix) {
        return this.rows === matrix.rows && this.columns === matrix.columns;
    }

    zeros() {
        this.entries.fill(0);

        return this;
    }

    ones() {
        this.entries.fill(1);

        return this;
    }

    scale(n = 1) {
        for (let i = 0; i < this.entries.length; i++) this.entries[i] *= n;

        return this;
    }

    static scale(matrix, n) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).scale(n);
    }

    add(n) {
        if (n instanceof Matrix && !this.isEqualShape(n)) throw new IllegalArgumentException("N must be a Matrix who's shape is equal to this Matrix");

        for (let i = 0; i < this.entries.length; i++) this.entries[i] += n instanceof Matrix ? n.entries[i] : n;

        return this;
    }

    static add(matrix, n) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).add(n);
    }

    sub(n) {
        if (n instanceof Matrix && !this.isEqualShape(n)) throw new IllegalArgumentException("N must be a Matrix who's shape is equal to this Matrix");

        for (let i = 0; i < this.entries.length; i++) this.entries[i] -= n instanceof Matrix ? n.entries[i] : n;

        return this;
    }

    static sub(matrix, n) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).sub(n);
    }

    multiply(matrix) {
        matrix = Matrix.multiply(this, matrix);
        this.entries = matrix.entries;
        this.buffer = matrix.buffer;
        this.columns = matrix.columns;

        return this;
    }

    static multiply(a, b) {
        if (!(a instanceof Matrix && b instanceof Matrix)) throw new IllegalArgumentException('A and b must be instances of Matrix');
        if (a.columns !== b.rows) throw new IllegalArgumentException("A must be a Matrix who's columns are equal to the rows of a Matrix B");

        const result = new Matrix(a.rows, b.columns);

        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.columns; j++) {
                let sum = 0;

                for (let k = 0; k < a.columns; k++) {
                    sum += a.entries[i * a.columns + k] * b.entries[k * b.columns + j];
                }

                result.entries[i * b.columns + j] = sum;
            }
        }

        return result;
    }

    product(matrix) {
        matrix = Matrix.product(this, matrix);
        this.entries = matrix.entries;
        this.buffer = matrix.buffer;

        return this;
    }

    static product(a, b) {
        if (!(a instanceof Matrix && b instanceof Matrix)) throw new IllegalArgumentException('A and b must be instances of Matrix');
        if (!a.isEqualShape(b)) throw new IllegalArgumentException("A must be a Matrix who's shape is equal to a Matrix B");

        const result = new Matrix(a);
        for (let i = 0; i < a.entries.length; i++) {
            result.entries[i] *= b.entries[i];
        }

        return result;
    }

    convolve(kernel, stride = 1, padding = 0) {
        const matrix = Matrix.convolve(this, kernel, stride, padding);
        this.entries = matrix.entries;
        this.buffer = matrix.buffer;
        this.rows = matrix.rows;
        this.columns = matrix.columns;

        return this;
    }

    static convolve(matrix, kernel, stride = 1, padding = 0) {
        if (stride < 1 || !Number.isInteger(stride)) throw new IllegalArgumentException('Stride must be an integer larger than 0');
        if (padding < 0 || !Number.isInteger(padding)) throw new IllegalArgumentException('Padding must be an non-negative integer');
        if (!(matrix instanceof Matrix && kernel instanceof Matrix)) throw new IllegalArgumentException('Matrix and kernel must be instances of Matrix');
        if (matrix.rows + padding < kernel.rows || matrix.columns + padding < kernel.columns) throw new IllegalArgumentException('Matrix shape cannot be smaller than kernel shape');

        const rows = (matrix.rows + padding * 2 - kernel.rows) / stride + 1,
            cols = (matrix.columns + padding * 2 - kernel.columns) / stride + 1;

        const result = new Matrix(Math.ceil(rows), Math.ceil(cols));
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.columns; j++) {

                let sum = 0;
                for (let k = 0; k < kernel.rows; k++) {
                    for (let l = 0; l < kernel.columns; l++) {
                        const rowIdx = i - padding > rows ? i - padding - 1 : i - padding,
                            colIdx = j - padding > cols ? j - padding - 1 : j - padding;

                        let index = (rowIdx + k) * matrix.columns + (colIdx * stride + l);
                        if (index < 0 || index >= matrix.entries.length) index = -1;

                        sum += kernel.entries[k * kernel.columns + l] * (index < 0 ? 0 : matrix.entries[index]);
                    }
                }

                result.entries[i * result.columns + j] = sum;
            }
        }

        return result;
    }

    transpose() {
        const matrix = Matrix.transpose(this);
        this.entries = matrix.entries;
        this.buffer = matrix.buffer;
        this.rows = matrix.rows;
        this.columns = matrix.columns;

        return this;
    }

    static transpose(matrix) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        const result = new Matrix(matrix.columns, matrix.rows);

        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.columns; j++) {
                result.entries[j * matrix.rows + i] = matrix.entries[i * matrix.columns + j];
            }
        }

        return result;
    }

    flip() {
        this.entries.reverse();

        return this;
    }

    static flip(matrix) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).flip();
    }

    transform(f) {
        if (!(f instanceof Function)) throw new IllegalArgumentException('F must be an instance of Function');

        for (let i = 0; i < this.entries.length; i++) this.entries[i] = f(this.entries[i]);

        return this;
    }

    static transform(matrix, f) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).transform(f);
    }

    reshape(rows, columns = rows) {
        if (rows * columns !== this.entries.length) throw new IllegalArgumentException('New matrix size must be equal to previous size');

        this.rows = rows;
        this.columns = columns;

        return this;
    }

    static reshape(matrix, rows, columns = rows) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).reshape(rows, columns);
    }

    flat() {
        return this.reshape(this.entries.length, 1);
    }

    static flat(matrix) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).flat();
    }

    clip(min = 0, max = 1) {
        for (let i = 0; i < this.entries.length; i++) {
            if (min !== null) this.entries[i] = this.entries[i] < min ? min : this.entries[i];
            if (max !== null) this.entries[i] = this.entries[i] > max ? max : this.entries[i];
        }

        return this;
    }

    static clip(matrix, min = 0, max = 1) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).clip(min, max);
    }

    static identity(n) {
        const matrix = new Matrix(n);

        for (let i = 0; i < n; i++) {
            matrix.entries[n + i] = 1;
        }

        return matrix;
    }

    static random(rows, columns = rows, min = 0, max = 1, decimals = -1) {
        const matrix = new Matrix(rows, columns);

        for (let i = 0; i < matrix.entries.length; i++) {
            matrix.entries[i] = Math.random() * (max - min) + min;
            if (decimals >= 0) matrix.entries[i] = Math.round(matrix.entries[i] * Math.pow(10, decimals)) / Math.pow(10, decimals);
        }

        return matrix;
    }

    sum() {
        return this.entries.reduce((a, b) => a + b, 0);
    }

    static fromArray(array) { //clone data
        if (!Array.isArray(array)) throw new IllegalArgumentException('Array must be an instance of Array');

        return new Matrix(array.length, 1, array);
    }

    toArray() { //maybe clone data
        return this.entries;
    }

    serialize() {
        return { ...this };
    }

    static deserialize(data) {
        return new Matrix(data.rows, data.columns, data.entries);
    }

};