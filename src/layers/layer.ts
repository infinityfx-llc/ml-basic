import { Activator, Initialization, Initializer, Initializers } from "../lib/functions";
import Matrix from "../lib/matrix";
import GradientDescent from "../optimizers/gradient-descent";
import Optimizer from "../optimizers/optimizer";

export default abstract class Layer {

    type = 'Layer';
    abstract name: string;
    input: [number, number];
    output: [number, number];
    activation: Activator;
    initializer: Initializer;
    optimizer: Optimizer = new GradientDescent();

    constructor(input: [number, number], output: [number, number], activation: Activator, initialization: Initialization = 'uniform') {
        this.input = input;
        this.output = output;
        this.activation = activation;
        this.initializer = Initializers[initialization];
    }

    abstract propagate(input: Matrix): Matrix;

    abstract backPropagate(input: Matrix, output: Matrix, loss: Matrix): Matrix;

}