import Exception from '../exceptions/exception';

export default class Loss {
    
    static name = 'identity';

    constructor() {
        throw new Exception();
    }

    static serialize() {
        return this.name;
    }

}