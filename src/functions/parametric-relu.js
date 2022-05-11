import Activator from './activator';

export default class ParametricRelu extends Activator {

    static name = 'parametric_relu';
    static alpha = 0.01;

    static function(n) {
        return n < 0 ? ParametricRelu.alpha * n : n;
    }

    static derivative(n) {
        return n < 0 ? ParametricRelu.alpha : 1;
    }

}