module.exports = class Exception extends Error {

    constructor(message) {
        super(message);
        this.name = 'Exception';
    }

};