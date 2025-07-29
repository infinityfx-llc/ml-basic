import Matrix from "../lib/matrix";

export type HyperParameters = {
    learningRate?: number;
    clipping?: number;
    batchSize?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
};

export default abstract class Optimizer {
    
    type = 'Optimizer';
    abstract name: string;
    abstract step(input: Matrix, gradient: Matrix, callback: (input: Matrix, gradient: Matrix) => void): void;

    clone() {
        return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
    }

    configure(options: HyperParameters) {
        Object.assign(this, options);
    }

}