import Activator from './activator';

export default class TanH extends Activator {

    static name = 'tanh';

    static function(n) {
        return Math.tanh(n);
    }

    static derivative(n) {
        return 1 - (n * n);
    }

}