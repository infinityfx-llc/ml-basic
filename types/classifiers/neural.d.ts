import { Classifier } from '../classifiers/classifier';
import { Matrix } from '../math/matrix';
import { HyperParameters } from '../global';

export class Neural extends Classifier {

    constructor({ shape, optimizer, loss_function, hyper_parameters, options }?: { 
        shape?: number[];
        optimizer?: string;
        loss_function?: string;
        hyper_parameters?: HyperParameters;
        options?: Classifier.Options;
    });

    propagate(input: number[]): Promise<Matrix[]>;

    predict(input: number[]): Promise<number[]> | {
        label: string;
        certainty: number;
    };

    backPropagate(input: number[], target: number[], hyper_parameters?: HyperParameters): Promise<number>;

    flush(): void;

    fit(data: {
        input: number[];
        target: number[];
    }[], { max_epochs, error_threshold, iterative, hyper_parameters }?: {
        max_epochs?: number;
        error_threshold?: number;
        iterative?: boolean;
        hyper_parameters?: HyperParameters;
    }): Promise<Number>;

    tune(data: {
        input: number[];
        target: number[];
    }[], parameters?: object, { max_epochs }?: {
        max_epochs?: number;
    }): Promise<object>;
    
    loadModel(model: object): Classifier;

    serialize(): string;

}