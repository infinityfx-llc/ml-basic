import Activator from './activator';

export default class Elu extends Activator {

    static name = 'elu';
    static alpha = 1;

    static function(n) {
        return n > 0 ? n : Elu.alpha * (Math.exp(n) - 1);
    }

    static derivative(n) {
        return n < 0 ? Elu.alpha * Math.exp(n) : 1;
    }

}