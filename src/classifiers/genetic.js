const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Layer = require('../layer');
const { argmin, combine } = require('../utils');
const Classifier = require('./classifier');

module.exports = class Genetic extends Classifier {

    static name = 'genetic';

    constructor({ shape = [2, 1], fitness_function = null, hyper_parameters = {}, options = {} } = {}) {
        super(options.multithreaded);

        this.generations = 0;
        this.fittest = 0;

        this.candidate_threshold = hyper_parameters.candidate_threshold || 0.1;
        if (fitness_function) {
            if (!(fitness_function instanceof Function)) throw new IllegalArgumentException('Fitness function must be an instance of Function');
            this.fitness = fitness_function;
        }

        this.population_size = hyper_parameters.population_size || 20;
        this.population = new Array(this.population_size).fill(0).map(() => {
            const [network, __shape] = this.createNetwork(shape);
            this.shape = __shape;

            return network;
        });
    }

    async predict(input, candidate = this.fittest) {
        const outputs = await Classifier.propagate({
            input,
            network: this.population[candidate],
            input_size: this.shape.input
        });
        
        return outputs[outputs.length - 1].toArray();
    }

    async fitness() {
        throw new Exception('Please define a fitness function');
    }

    flush() {
        this.population.forEach(network => network.forEach(layer => layer.flush()));
        this.generations = 0;
        this.fittest = 0;
    }

    async evaluate() {
        const candidates = new Array(this.population_size);

        for (let i = 0; i < this.population_size; i++) {
            candidates[i] = await this.fitness(async input => await this.predict(input, i));
        }

        return candidates;
    }

    async evolve({ generations = 1, hyper_parameters = {} } = {}) {
        let fitness;

        for (let i = 0; i < generations; i++) {
            let candidates = combine(await this.evaluate(), this.population);

            const count = Math.ceil(this.population_size * (hyper_parameters.candidate_threshold || this.candidate_threshold));
            candidates = candidates.sort((a, b) => b[0] - a[0]).slice(0, count).map(([_, network]) => network);
            this.cross(candidates, hyper_parameters);

            const validation = await this.evaluate();
            this.fittest = argmin(validation);
            fitness = validation[this.fittest];

            this.generations++;
        }

        return fitness;
    }

    async tune(parameters = {}, { generations = 100 } = {}) {
        return await super.tune(async (hyper_parameters) => {
            const fitness = await this.evolve({
                generations,
                hyper_parameters
            });

            return 1 - fitness;
        }, parameters);
    }

    //implement different optimizers (decreasing mutation rate)
    async cross(candidates, hyper_parameters) {
        for (let i = 0; i < this.population_size; i++) {
            const a = candidates[Math.floor(Math.random() * candidates.length)];
            const b = candidates[Math.floor(Math.random() * candidates.length)];

            for (let j = 0; j < this.population[i].length; j++) {
                this.population[i][j] = Layer.cross(a[j], b[j]);
                this.population[i][j].mutate(hyper_parameters);
            }
        }
    }

    loadModel(model) {
        Object.entries(model).forEach(([key, val]) => {
            if (key === 'population') return this[key] = val.map(network => {
                network.map(layer => Layer.deserialize(layer));
            });

            this[key] = val;
        });
        this.shape.input = this.shape[0];
        this.shape.output = this.shape[this.shape.length - 1];

        return this;
    }

    serialize() {
        return JSON.stringify(Object.assign({ name: this.__proto__.constructor.name }, this), (key, value) => {
            if (value instanceof Layer) return value.serialize();
            if (value instanceof Pool || key === 'multithreading') return;
            if (value instanceof Float64Array) return Array.prototype.slice.call(value);

            return value;
        }, '\t');
    }

};