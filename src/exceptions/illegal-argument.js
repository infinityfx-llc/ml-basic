import Exception from './exception';

export default class IllegalArgumentException extends Exception {

    constructor(message) {
        super(message);
    }

}