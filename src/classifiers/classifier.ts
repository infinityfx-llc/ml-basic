import Matrix from "../lib/matrix";
import { saveJsonFile } from "../lib/utils";

export default abstract class Classifier {

    abstract name: string;

    abstract propagate(input: Matrix): Matrix;

    abstract backPropagate(input: Matrix, target: Matrix): number;

    export() {
        return JSON.stringify(this, (_, value) => {
            return typeof value === 'object' && 'serialize' in value && value.serialize instanceof Function ?
                value.serialize() :
                value;
        });
    }

    save(file: string) {
        saveJsonFile(file, this.export());
    }

}