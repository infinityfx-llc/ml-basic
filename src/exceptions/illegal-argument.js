const Exception = require('./exception');

module.exports = class IllegalArgumentException extends Exception {

    constructor(message) {
        super(message);
        this.name = 'IllegalArgumentException';
    }

};