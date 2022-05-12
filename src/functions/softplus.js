const Activator = require('./activator');

module.exports = class Softplus extends Activator {

    static name = 'softplus';

    static function(n) {
        return Math.log(1 + Math.exp(n));
    }

    static derivative(n) {
        return 1 / (1 + Math.exp(-n));
    }

};