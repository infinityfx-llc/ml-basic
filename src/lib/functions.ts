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
        return new Matrix(target).sub(output).apply(val => val * val).mean();
    }

    derivative(output: Matrix, target: Matrix) {
        return new Matrix(target).sub(output).apply(val => -2 * val / target.entries.length);
    }

}

export class CrossEntropyLoss extends LossFunction {

    name = 'CrossEntropyLoss';
    eps = 1e-15;

    mean(output: Matrix, target: Matrix) {

        return -new Matrix(output)
            .apply(val => Math.log(Math.min(Math.max(val, this.eps), 1 - this.eps)))
            .scale(target)
            .mean();
    }

    derivative(output: Matrix, target: Matrix) {
        const scaled = new Matrix(output).apply(y => {
            const clampedY = Math.min(Math.max(y, this.eps), 1 - this.eps);

            return 1 / (Math.max(clampedY * (1 - clampedY), this.eps) * target.entries.length);
        });

        return new Matrix(output)
            .sub(target)
            .scale(scaled);
    }

}

export abstract class Activator {

    abstract name: string;

    abstract activate(input: Matrix): Matrix;

    abstract deactivate(output: Matrix): Matrix;

    abstract derivative(output: Matrix, loss: Matrix): Matrix;

    serialize() {
        return this.name;
    }

}

export class Sigmoid extends Activator {

    name = 'Sigmoid';

    activate(input: Matrix) {
        return input.apply(n => 1 / (1 + Math.exp(-n)));
    }

    deactivate(output: Matrix) {
        return output.apply(n => n * (1 - n));
    }

    derivative(output: Matrix, loss: Matrix) {
        return this.deactivate(output).scale(loss);
    }

}

export class TanH extends Activator {

    name = 'TanH';

    activate(input: Matrix) {
        return input.apply(Math.tanh);
    }

    deactivate(output: Matrix) {
        return output.apply(n => 1 - n * n);
    }

    derivative(output: Matrix, loss: Matrix) {
        return this.deactivate(output).scale(loss);
    }

}

export class Elu extends Activator {

    name = 'Elu';
    alpha: number;

    constructor(alpha = 1) {
        super();

        this.alpha = alpha;
    }

    activate(input: Matrix) {
        return input.apply(n => n > 0 ? n : this.alpha * (Math.exp(n) - 1));
    }

    deactivate(output: Matrix) {
        return output.apply(n => n < 0 ? n + this.alpha : 1);
    }

    derivative(output: Matrix, loss: Matrix) {
        return this.deactivate(output).scale(loss);
    }

}

export class Relu extends Elu {

    name = 'Relu';

    constructor(alpha = 0.01) {
        super(alpha);
    }

    activate(input: Matrix) {
        return input.apply(n => n < 0 ? this.alpha * n : n);
    }

    deactivate(output: Matrix) {
        return output.apply(n => n < 0 ? this.alpha : 1);
    }

    derivative(output: Matrix, loss: Matrix) {
        return this.deactivate(output).scale(loss);
    }

}

export class SoftPlus extends Activator {

    name = 'SoftPlus';

    activate(input: Matrix) {
        return input.apply(n => Math.log(1 + Math.exp(n)));
    }

    deactivate(output: Matrix) {
        return output.apply(n => 1 - Math.exp(-n));
    }

    derivative(output: Matrix, loss: Matrix) {
        return this.deactivate(output).scale(loss);
    }

}

export class Gelu extends Activator {

    name = 'Gelu';

    activate(input: Matrix) {
        return input.apply(n => 0.5 * n * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (n + 0.044715 * Math.pow(n, 3)))));
    }

    deactivate(output: Matrix) {
        const c = Math.sqrt(2 / Math.PI);

        return output.apply(n => {
            const t = Math.pow(Math.tanh(c * (n + 0.044715 * Math.pow(n, 3))), 2);
            return 0.5 * (1 + t) + 0.5 * n * (1 - t * t) * (c * (1 + 3 * 0.044715 * n * n));
        });
    }

    derivative(output: Matrix, loss: Matrix) {
        return this.deactivate(output).scale(loss);
    }

}

export class Softmax extends Activator {

    name = 'Softmax';

    activate(input: Matrix) {
        return input.sub(input.max()).apply(Math.exp).scale(1 / (input.sum() || 1));
    }

    deactivate(output: Matrix) {
        return output.apply(n => n * (1 - n));
    }

    derivative(output: Matrix, loss: Matrix) {
        return new Matrix(loss).sub(loss.dot(output)).scale(output);
    }

}

export type Initializer = (row: number, cols: number) => Matrix;

const xavier: Initializer = (rows, cols) => {
    const limit = Math.sqrt(6 / (cols + rows));

    return Matrix.random(rows, cols, -limit, limit);
}

const kaiming: Initializer = (rows, cols) => {
    const limit = Math.sqrt(6 / cols);

    return Matrix.random(rows, cols, -limit, limit);
}

const uniform: Initializer = (rows, cols) => Matrix.random(rows, cols, -1, 1);

export const Initializers = {
    xavier,
    kaiming,
    uniform
};

export type Initialization = keyof typeof Initializers;