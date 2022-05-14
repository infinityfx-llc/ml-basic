export as namespace MLBasic;

declare interface HyperParameters {
    learning_rate?: number;
}

export class Layer {}

export abstract class Classifier {
    
    private parseShape;

    private createNetwork;

    abstract predict(network: Layer[], input: number[]): Promise<number[]>;

    abstract tune(iteration_function: (hyper_parameters: HyperParameters) => Promise<number>, parameters: object): Promise<object>;

    export(path: string): void;

}

export class Neural extends Classifier {

    constructor({ shape, optimizer, loss_function, hyper_parameters }?: { 
        shape?: number[];
        optimizer?: string;
        loss_function?: string;
        hyper_parameters?: HyperParameters
    });

    predict(input: number[]): Promise<number[]> | {
        label: string;
        certainty: number;
    };

    private backPropagate;

    flush(): void;

    fit(data: {
        input: number[];
        target: number[];
    }[], { max_epochs, error_threshold, iterative, hyper_parameters }?: {
        max_epochs?: number;
        error_threshold?: number;
        iterative?: boolean;
        hyper_parameters?: HyperParameters
    }): Promise<Number>;

    tune(data: {
        input: number[];
        target: number[];
    }[], parameters?: object, { max_epochs }?: {
        max_epochs?: number
    }): Promise<object>;
    
    loadModel(model: object): Classifier;

    serialize(): string;

}

export function fromFile(path: string): Promise<Classifier>;

export function fromFile(file: File): Promise<Classifier>;

export class PreProcessor {

    private static inputKeys;
    private static targetKeys;

    constructor(data: {
        input?: number | number[];
        target?: number | number[];
    });

    private findKey;

    private hash;

    clean({ nullToZero, removeDuplicates }?: {
        nullToZero?: boolean;
        removeDuplicates?: boolean;
    }): PreProcessor;

    normalize(min?: number, max?: number): PreProcessor;

    min(key?: string): number;

    max(key?: string): number;

    split(): void;

    out(): {
        input: number[];
        target: number[];
    };

}