import Matrix from '../math/matrix';
import Loss from './loss';

export default class SquaredLoss extends Loss {

    static name = 'squared_loss';

    static mean(output, target) {
        return Matrix.sub(target, output).transform((val) => val * val).sum() / target.entries.length;
    }

    static derivative(output, target) {
        return Matrix.sub(target, output);
    }

}