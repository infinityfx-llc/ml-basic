import Activator from './activator';

export default class Sigmoid extends Activator {

    static name = 'sigmoid';

    static function(n) {
        return 1 / (1 + Math.exp(-n));
    }

    static derivative(n) {
        return n * (1 - n);
    }

}