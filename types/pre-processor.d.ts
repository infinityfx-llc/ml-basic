declare interface PreProcessor {
    data: {
        input: number[];
        target: number[];
    }[];

    private labels;

    private shape;
}

export class PreProcessor {

    private static inputKeys;

    private static targetKeys;

    constructor(data: PreProcessor.Data[], target?: PreProcessor.Data[]);

    private fromLabel;

    private toArray;

    private findKey;

    private hash;

    private trackSize;

    clean({ nullToZero, removeDuplicates, allowVariableData }?: {
        nullToZero?: boolean;
        removeDuplicates?: boolean;
        allowVariableData?: boolean;
    }): PreProcessor;

    normalize(min?: number, max?: number): PreProcessor;

    min(key?: string): number;

    max(key?: string): number;

    split(): void;

    out(): {
        input: number[];
        target: number[];
    };

    getLabels(): string[];

}

export namespace PreProcessor {

    type Data =
        number |
        string |
        [
            number | number[],
            number | number[] | string
        ] |
        {
            input?: number | number[];
            in?: number | number[];
            data?: number | number[];
            target?: number | number[] | string;
            output?: number | number[] | string;
            out?: number | number[] | string;
            label?: number | number[] | string;
        };

}