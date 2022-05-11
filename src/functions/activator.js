import Exception from '../exceptions/exception';

export default class Activator {

    static name = 'identity';

    constructor() {
        throw new Exception();
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

}