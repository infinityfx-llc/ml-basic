declare interface Log {
    elapsed: number;
    entries: {
        any: any;
        timestamp: number;
    }[];
}

export class Log {

    constructor();

    add(entry?: object): Log;

    set(attributes?: object): Log;

    increment(attributes?: object): Log;

    end(): Log;

}