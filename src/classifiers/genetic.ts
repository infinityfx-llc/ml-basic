import Matrix from "../lib/matrix";
import BatchGradientDescent from "../optimizers/batch-gradient-descent";
import Optimizer from "../optimizers/optimizer";
import Classifier from "./classifier";
import Neural from "./neural";

export type Individual<O extends Optimizer> = {
    model: Neural<O>;
    fitness: number;
};

export default class Genetic<O extends Optimizer = BatchGradientDescent> extends Classifier {

    name = 'Genetic';
    baseModel: Neural<O>;
    mutationRate: number;
    mutationScale: number;
    elitism: number;
    generation = 0;
    population: Individual<O>[];
    champion: Individual<O>;

    constructor({
        baseModel,
        populationSize = 20,
        mutationRate = 0.05,
        mutationScale = 0.1,
        elitism = 1
    }: {
        baseModel: Neural<O>;
        populationSize?: number;
        mutationRate?: number;
        mutationScale?: number;
        elitism?: number;
    }) {
        super();

        this.baseModel = baseModel;
        this.mutationRate = mutationRate;
        this.mutationScale = mutationScale;
        this.elitism = Math.min(elitism, populationSize);

        this.population = new Array(populationSize).fill(0).map(() => ({
            model: baseModel.clone(),
            fitness: -Infinity
        }));

        this.champion = this.population[0];
    }

    private selectParent(tournamentSize = 3): Neural<any> {
        let best = this.population[Math.floor(Math.random() * this.population.length)];

        for (let i = 1; i < tournamentSize; i++) {
            const candidate = this.population[Math.floor(Math.random() * this.population.length)];
            if (candidate.fitness > best.fitness) {
                best = candidate;
            }
        }

        return best.model;
    }

    evolve({
        generations,
        fitness,
        onGeneration,
        targetFitness = Infinity
    }: {
        generations: number;
        fitness: (model: Neural<any>, index: number) => number | Promise<number>;
        onGeneration?: (stats: { generation: number; bestFitness: number; avgFitness: number; bestModel: Neural<any> }) => void | Promise<void>;
        targetFitness?: number;
    }) {
        const runGeneration = async () => {
            let totalFitness = 0;

            for (let i = 0; i < this.population.length; i++) {
                const fit = await fitness(this.population[i].model, i);
                this.population[i].fitness = fit;
                totalFitness += fit;

                if (fit > this.champion.fitness) {
                    this.champion.fitness = fit;
                    this.champion.model = this.population[i].model;
                }
            }

            this.population.sort((a, b) => b.fitness - a.fitness);

            const avgFitness = totalFitness / this.population.length;

            await onGeneration?.({
                generation: this.generation,
                bestFitness: this.population[0].fitness,
                avgFitness,
                bestModel: this.champion.model
            });

            if (this.population[0].fitness >= targetFitness) return true;

            const nextPopulation: Individual<O>[] = new Array(this.population.length);

            for (let i = 0; i < this.elitism; i++) {
                nextPopulation[i] = this.population[i];
            }

            // Fill remainder with crossover + mutation
            for (let i = this.elitism; i < this.population.length; i++) {
                const parentA = this.selectParent();
                const parentB = this.selectParent();

                nextPopulation[i] = {
                    model: parentA.cross(parentB).mutate(this.mutationRate, this.mutationScale),
                    fitness: -Infinity
                };
            }

            this.population = nextPopulation;
            this.generation++;

            return false;
        };

        return (async () => {
            for (let g = 0; g < generations; g++) {
                const done = await runGeneration();
                if (done) break;
            }

            return this.champion.model;
        })();
    }

    propagate(input: Matrix) {
        return this.champion.model.propagate(input);
    }

    backPropagate(): number {
        throw new Error('Genetic classifier does not use gradient backpropagation; use evolve() instead');
    }

    predict(options: Parameters<Neural<O>['predict']>[0]) {
        return this.champion.model.predict(options);
    }

}