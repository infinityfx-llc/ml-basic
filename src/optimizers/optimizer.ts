import Matrix from "../lib/matrix";

export default abstract class Optimizer {
    
    abstract name: string;
    abstract step(gradient: Matrix, batching?: boolean): Matrix;

    clone() {
        return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
    }

    configure(options: {
        learningRate?: number;
        clipping?: number;
        batchSize?: number;
        beta1?: number;
        beta2?: number;
        epsilon?: number;
    }) {
        Object.assign(this, options);
    }

}