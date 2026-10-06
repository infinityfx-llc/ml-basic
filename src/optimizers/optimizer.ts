import Matrix from "../lib/matrix";

export type HyperParameters = {
    learningRate?: number;
    clipping?: number;
    batchSize?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
};

export type ParameterUpdate = {
    param: Matrix;
    gradient: Matrix;
};

export default abstract class Optimizer {
    
    type = 'Optimizer';
    abstract name: string;
    abstract step(updates: ParameterUpdate[]): void;

    flush?(): void;

    abstract clone(): this;

    configure(options: HyperParameters) {
        Object.assign(this, options);
    }

}