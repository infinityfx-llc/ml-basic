export class Pool {

    constructor(max_threads?: number);

    private onfreethread;

    private addThread;

    queue(task: object): Promise<object>;

    flush(): void;

}