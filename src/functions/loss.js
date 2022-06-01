const Exception = require('../exceptions/exception');

module.exports = class Loss {
    
    static name = 'identity';

    constructor() {
        throw new Exception('Cannot instantiate abstract class Loss');
    }

    static serialize() {
        return this.name;
    }

};