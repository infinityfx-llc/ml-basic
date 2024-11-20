import Matrix from "../lib/matrix";

export default abstract class Optimizer {

    abstract step(gradient: Matrix): Matrix;

    clone() {
        return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
    }

}