import Matrix from "./matrix";

function argmax(matrix: Matrix): number {
    let maxIdx = 0;
    let maxVal = matrix.entries[0];

    for (let i = 1; i < matrix.entries.length; i++) {
        if (matrix.entries[i] > maxVal) {
            maxVal = matrix.entries[i];
            maxIdx = i;
        }
    }

    return maxIdx;
}

function getClass(matrix: Matrix): number {
    if (matrix.entries.length === 1) {
        return matrix.entries[0] >= 0.5 ? 1 : 0;
    }

    return argmax(matrix);
}

export abstract class Metric<T = any> {
    abstract name: string;
    abstract compute(predictions: Matrix[], targets: Matrix[]): T;
}

export class Accuracy extends Metric {

    name = 'accuracy';

    compute(predictions: Matrix[], targets: Matrix[]): number {
        if (predictions.length === 0 || predictions.length !== targets.length) {
            throw new Error('Predictions and targets must be non-empty and have matching lengths');
        }

        let correct = 0;

        for (let i = 0; i < predictions.length; i++) {
            if (getClass(predictions[i]) === getClass(targets[i])) {
                correct++;
            }
        }

        return correct / predictions.length;
    }

}

export class ConfusionMatrix extends Metric {

    name = 'confusionMatrix';

    compute(predictions: Matrix[], targets: Matrix[]) {
        if (predictions.length === 0 || predictions.length !== targets.length) {
            throw new Error('Predictions and targets must be non-empty and have matching lengths');
        }

        let maxClass = 1;
        for (let i = 0; i < predictions.length; i++) {
            maxClass = Math.max(maxClass, getClass(predictions[i]), getClass(targets[i]));
        }
        const numClasses = maxClass + 1;

        const matrix = new Matrix(numClasses, numClasses);

        for (let i = 0; i < predictions.length; i++) {
            const actual = getClass(targets[i]);
            const predicted = getClass(predictions[i]);

            if (actual < numClasses && predicted < numClasses) {
                matrix.entries[actual * numClasses + predicted]++;
            }
        }

        return matrix;
    }

}

export class Precision extends Metric {

    name = 'precision';

    compute(predictions: Matrix[], targets: Matrix[]) {
        const cm = new ConfusionMatrix().compute(predictions, targets);
        const k = cm.rows;
        let sumPrecision = 0;
        let validClasses = 0;

        for (let c = 0; c < k; c++) {
            const tp = cm.entries[c * k + c];
            let predictedCount = 0;

            for (let r = 0; r < k; r++) {
                predictedCount += cm.entries[r * k + c];
            }

            if (predictedCount > 0) {
                sumPrecision += tp / predictedCount;
                validClasses++;
            }
        }

        return validClasses > 0 ? sumPrecision / validClasses : 0;
    }

}

export class Recall extends Metric {

    name = 'recall';

    compute(predictions: Matrix[], targets: Matrix[]) {
        const cm = new ConfusionMatrix().compute(predictions, targets);
        const k = cm.rows;
        let sumRecall = 0;
        let validClasses = 0;

        for (let c = 0; c < k; c++) {
            const tp = cm.entries[c * k + c];
            let actualCount = 0;

            for (let j = 0; j < k; j++) {
                actualCount += cm.entries[c * k + j];
            }

            if (actualCount > 0) {
                sumRecall += tp / actualCount;
                validClasses++;
            }
        }

        return validClasses > 0 ? sumRecall / validClasses : 0;
    }

}

export class F1Score extends Metric {

    name = 'f1Score';

    compute(predictions: Matrix[], targets: Matrix[]) {
        const p = new Precision().compute(predictions, targets);
        const r = new Recall().compute(predictions, targets);

        if (p + r === 0) return 0;

        return (2 * p * r) / (p + r);
    }

}

export class MeanAbsoluteError extends Metric {

    name = 'mae';

    compute(predictions: Matrix[], targets: Matrix[]) {
        if (predictions.length === 0 || predictions.length !== targets.length) return 0;

        let totalError = 0;
        let totalCount = 0;

        for (let i = 0; i < predictions.length; i++) {
            totalError += new Matrix(predictions[i]).sub(targets[i]).apply(Math.abs).sum();
            totalCount += predictions[i].entries.length;
        }

        return totalError / totalCount;
    }

}

export class MeanSquareError extends Metric {

    name = 'mse';

    compute(predictions: Matrix[], targets: Matrix[]) {
        if (predictions.length === 0 || predictions.length !== targets.length) return 0;

        let totalSqError = 0;
        let totalCount = 0;

        for (let i = 0; i < predictions.length; i++) {
            const d = predictions[i].dist(targets[i]);
            totalSqError += d * d;
            totalCount += predictions[i].entries.length;
        }

        return totalSqError / totalCount;
    }

}

export class RootMeanSquareError extends MeanSquareError {

    name = 'rmse';

    compute(predictions: Matrix[], targets: Matrix[]) {
        return Math.sqrt(super.compute(predictions, targets));
    }

}

export class R2Score extends Metric {

    name = 'r2Score';

    compute(predictions: Matrix[], targets: Matrix[]) {
        if (predictions.length === 0 || predictions.length !== targets.length) return 0;

        let sumTarget = 0;
        let totalCount = 0;

        for (const t of targets) {
            sumTarget += t.sum();
            totalCount += t.entries.length;
        }

        const meanTarget = sumTarget / totalCount;
        let ssRes = 0;
        let ssTot = 0;

        for (let i = 0; i < predictions.length; i++) {
            const d = predictions[i].dist(targets[i]);
            ssRes += d * d;

            ssTot += new Matrix(targets[i]).apply(val => {
                const diff = val - meanTarget;
                return diff * diff;
            }).sum();
        }

        return 1 - (ssRes / ssTot);
    }

}

export const Metrics = {
    accuracy: () => new Accuracy(),
    precision: () => new Precision(),
    recall: () => new Recall(),
    f1Score: () => new F1Score(),
    confusionMatrix: () => new ConfusionMatrix(),
    mae: () => new MeanAbsoluteError(),
    mse: () => new MeanSquareError(),
    rmse: () => new RootMeanSquareError(),
    r2Score: () => new R2Score()
};

export default Metrics;
