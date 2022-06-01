import { Classifier } from '../classifiers/classifier';
import { Matrix } from '../math/matrix';
import { HyperParameters, Loss, Optimizer } from '../global';
import { Log } from '../log';
import { Layer } from '../layers/layer';

declare interface Neural implements Classifier {
    epochs: number;
    error: number;
    loss: object;
    layers: Layer[];
    shape: number[];
    labels: string[] | null;
    binary: boolean;
}

export class Neural extends Classifier {

    static name = 'neural';

    constructor({ shape, optimizer, loss_function, hyper_parameters, options }?: { 
        shape?: Classifier.Shape;
        optimizer?: Optimizer;
        loss_function?: Loss;
        hyper_parameters?: HyperParameters;
        options?: Classifier.Options;
    });

    static model(): Neural;

    add(layer: Layer): Neural;

    setLoss(loss_function?: Loss): Neural;

    compile({ binary, labels }?: {
        binary: boolean;
        labels: string[];
    }): Neural;

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
    }): Promise<Log>;

    tune(data: {
        input: number[];
        target: number[];
    }[], parameters?: object, { max_epochs }?: {
        max_epochs?: number;
    }): Promise<object>;
    
    loadModel(model: object): Classifier;

    serialize(): string;

}