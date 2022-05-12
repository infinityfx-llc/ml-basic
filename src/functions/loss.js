const Exception = require('../exceptions/exception');

module.exports = class Loss {
    
    static name = 'identity';

    constructor() {
        throw new Exception();
    }

    static serialize() {
        return this.name;
    }

};