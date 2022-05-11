import IllegalArgumentException from '../exceptions/illegal-argument';

export default class Matrix {

    constructor(rows, columns = rows, entries = []) {
        if (rows instanceof Matrix) return Matrix.copy(rows);
        if (rows < 1 || columns < 1) throw new IllegalArgumentException('Rows and Columns must be numbers greater than 0');

        this.rows = rows;
        this.columns = columns;
        this.entries = new Array(rows * columns).fill(0).map((val, i) => entries[i] || val);
    }

    static copy(matrix) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix.rows, matrix.columns, matrix.entries);
    }

    isEqualShape(matrix) {
        return this.rows === matrix.rows && this.columns === matrix.columns;
    }

    zeros() {
        this.entries = new Array(this.entries.length).fill(0);

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
        this.entries = Matrix.multiply(this, matrix).entries;
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
        this.entries = Matrix.product(this, matrix).entries;

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

    transpose() {
        const matrix = Matrix.transpose(this);
        this.entries = matrix.entries;
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

    transform(f) {
        if (!(f instanceof Function)) throw new IllegalArgumentException('F must be an instance of Function');

        for (let i = 0; i < this.entries.length; i++) this.entries[i] = f(this.entries[i]);

        return this;
    }

    static transform(matrix, f) {
        if (!(matrix instanceof Matrix)) throw new IllegalArgumentException('Matrix must be an instance of Matrix');

        return new Matrix(matrix).transform(f);
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

    static fromArray(array) {
        if (!Array.isArray(array)) throw new IllegalArgumentException('Array must be an instance of Array');

        return new Matrix(array.length, 1, array);
    }

    toArray() {
        return this.entries;
    }

    serialize() {
        return { ...this };
    }

    static deserialize(data) {
        return new Matrix(data.rows, data.columsn, data.entries);
    }

}