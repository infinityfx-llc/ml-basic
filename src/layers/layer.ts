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

    mutate(rate: number, scale: number) {
        for (const key in this) {
            if (this[key] instanceof Matrix) {
                const entries = this[key].entries;

                for (let i = 0; i < entries.length; i++) {
                    if (Math.random() < rate) {
                        entries[i] += (Math.random() * 2 - 1) * scale;
                    }
                }
            }
        }
    }

    cross(layer: Layer) {
        for (const key in this) {
            const entry = layer[key as keyof Layer];

            if (this[key] instanceof Matrix && entry instanceof Matrix) {
                if (!this[key].isEqualShape(entry)) throw new Error('Matrices must have an equal shape to compute discrete crossover');

                for (let i = 0; i < this[key].entries.length; i++) {
                    if (Math.random() < .5) this[key].entries[i] = entry.entries[i];
                }
            }
        }
    }

    clone() {
        const clone: Layer = Object.create(Object.getPrototypeOf(this));

        for (const key in this) {
            if (this[key] instanceof Matrix) {
                // @ts-expect-error
                clone[key] = new Matrix(this[key]);
            } else
                if (this[key] instanceof Optimizer) {
                    // @ts-expect-error
                    clone[key] = this.optimizer.clone();
                } else {
                    // @ts-expect-error
                    clone[key] = this[key];
                }
        }

        return clone;
    }

}