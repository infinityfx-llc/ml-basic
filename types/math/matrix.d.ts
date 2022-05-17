export class Matrix {

    constructor(rows: number, columns?: number, entries?: number[]);

    static copy(matrix: Matrix): Matrix;

    isEqualShape(matrix: Matrix): boolean;

    zeros(): Matrix;

    scale(n?: number): Matrix;

    static scale(matrix: Matrix, n: number): Matrix;

    add(n: number): Matrix;

    static add(matrix: Matrix, n: number): Matrix;

    sub(n: number): Matrix;

    static sub(matrix: Matrix, n: number): Matrix;

    multiply(matrix: Matrix): Matrix;

    static multiply(a: Matrix, b: Matrix): Matrix;

    product(matrix: Matrix): Matrix;

    static product(a: Matrix, b: Matrix): Matrix;

    transpose(): Matrix;

    static transpose(matrix: Matrix): Matrix;

    transform(f: (val: number) => number): Matrix;

    static transform(matrix: Matrix, f: (val: number) => number): Matrix;

    static identity(n: number): Matrix;

    static random(rows: number, columns?: number, min?: number, max?: number, decimals?: number): Matrix;

    sum(): number;

    static fromArray(array: number[]): Matrix;

    toArray(): number[];

    serialize(): object;

    static deserialize(data: object): Matrix;

}