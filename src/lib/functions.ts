import Matrix from "./matrix";

export abstract class LossFunction {

    abstract name: string;

    abstract mean(output: Matrix, target: Matrix): number;

    abstract derivative(output: Matrix, target: Matrix): Matrix;

    serialize() {
        return this.name;
    }

}

export class SquaredLoss extends LossFunction {

    name = 'SquaredLoss';

    mean(output: Matrix, target: Matrix) {
        return new Matrix(target).sub(output).apply(val => val * val).sum() / target.entries.length;
    }

    derivative(output: Matrix, target: Matrix) {
        return new Matrix(target).sub(output).apply(val => -2 * val / target.entries.length);
    }

}

export class CrossEntropyLoss extends LossFunction {

    name = 'CrossEntropyLoss';
    
    mean(output: Matrix, target: Matrix) { // check if output and target should be switched
        return -new Matrix(output).apply(Math.log).scale(target).sum();
    }

    derivative(output: Matrix, target: Matrix) {
        return new Matrix(output).sub(target);
    }

}

export abstract class Activator {

    abstract name: string;

    abstract activate(n: number): number;

    abstract deactivate(n: number): number;

    serialize() {
        return this.name;
    }

}

export class Sigmoid extends Activator {

    name = 'Sigmoid';

    activate(n: number) {
        return 1 / (1 + Math.exp(-n));
    }

    deactivate(n: number) {
        return n * (1 - n);
    }

}

export class TanH extends Activator {

    name = 'TanH';

    activate(n: number) {
        return Math.tanh(n);
    }

    deactivate(n: number) {
        return 1 - (n * n);
    }

}

export class Elu extends Activator {

    name = 'Elu';
    alpha: number;

    constructor(alpha = 1) {
        super();

        this.alpha = alpha;
    }

    activate(n: number) {
        return n > 0 ? n : this.alpha * (Math.exp(n) - 1);
    }

    deactivate(n: number) {
        return n < 0 ? this.alpha * Math.exp(n) : 1;
    }

}

export class Relu extends Elu {

    name = 'Relu';

    constructor(alpha = 0.01) {
        super(alpha);
    }

    activate(n: number) {
        return n < 0 ? this.alpha * n : n;
    }

    deactivate(n: number) {
        return n < 0 ? this.alpha : 1;
    }

}

export class SoftPlus extends Activator {

    name = 'SoftPlus';

    activate(n: number) {
        return Math.log(1 + Math.exp(n));
    }

    deactivate(n: number) {
        return 1 / (1 + Math.exp(-n));
    }

}