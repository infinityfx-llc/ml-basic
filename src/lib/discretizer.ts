import Matrix from "./matrix";

export abstract class Discretizer {

    keymap = new Map<string, number>();
    vocab = -1;

    abstract get(value: string): Matrix;

    constructor(vocabulary = -1) {
        this.keymap.set('<pad>', 0);
        this.vocab = vocabulary;
    }

    getVocabulary() {
        return Math.max(this.vocab, this.keymap.size);
    }

}

export class Labeler extends Discretizer {

    get(value: string) {
        let index = this.keymap.get(value);

        if (index === undefined) {
            this.keymap.set(value, index = this.keymap.size);
        }

        const matrix = new Matrix(Math.max(this.keymap.size, this.vocab), 1);
        matrix.entries[index] = 1;

        return matrix;
    }

}

export class Tokenizer extends Discretizer {

    get(value: string) {
        const tokens = value
            .toLocaleLowerCase()
            .split(/(\s+|[^\w\s])/g).map(token => { // contains spaces..
                let index = this.keymap.get(token);

                if (index === undefined) {
                    this.keymap.set(token, index = this.keymap.size);
                }

                return index;
            });

        return new Matrix(Math.max(tokens.length, this.vocab), 1, tokens);
    }

}