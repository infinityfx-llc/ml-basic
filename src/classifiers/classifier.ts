import Matrix from "../lib/matrix";

export default abstract class Classifier {

    abstract propagate(input: Matrix): Matrix;

    abstract backPropagate(input: Matrix, target: Matrix): number;

    export() {
        return JSON.stringify(this, (_, value) => {
            return typeof value === 'object' && 'serialize' in value && value.serialize instanceof Function ?
                value.serialize() :
                value;
        });
    }

}