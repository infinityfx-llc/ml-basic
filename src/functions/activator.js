const Exception = require('../exceptions/exception');

module.exports = class Activator {

    static name = 'identity';

    constructor() {
        throw new Exception('Cannot instantiate abstract class Activator');
    }

    static function(n) {
        return n;
    }

    static derivative(n) {
        return n;
    }

    static serialize() {
        return this.name;
    }

};