import GeneticFX from './classifiers/genetic';
import NeuralFX from './classifiers/neural';
import IllegalArgumentException from './exceptions/illegal-argument';

export const classifiers = {
    [NeuralFX.name]: NeuralFX,
    [GeneticFX.name]: GeneticFX
};

export const fromFile = async (file) => {
    if (!(file instanceof Blob)) throw new IllegalArgumentException('File must be an instance of File');

    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onload = () => {
            const model = JSON.parse(reader.result);
            const classifier = new classifiers[model.name]().loadModel(model);

            resolve(classifier);
        };
        reader.onerror = () => reject(new IllegalArgumentException('File must be an instance of File'));
        reader.readAsText(file);
    });
}