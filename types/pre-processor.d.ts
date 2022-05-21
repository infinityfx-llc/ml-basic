declare interface PreProcessor {
    data: {
        input: number[];
        target: number[];
    }[];
}

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