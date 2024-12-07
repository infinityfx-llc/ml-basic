import Matrix from "../lib/matrix";
import { browser } from "../lib/utils";
import { writeFile } from 'fs';

export default abstract class Classifier {

    abstract name: string;

    abstract propagate(input: Matrix): Matrix;

    abstract backPropagate(input: Matrix, target: Matrix): number;

    export() {
        return JSON.stringify(this, (_, value) => {
            return typeof value === 'object' && 'serialize' in value && value.serialize instanceof Function ?
                value.serialize() :
                value;
        });
    }

    save(file: string) {
        const data = this.export();

        if (!/\.json$/i.test(file)) file = file + '.json';
        if (browser()) {
            file = file.replace(/.*\//, '');

            const blob = new Blob([data], { type: 'application/json' }),
                a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = file;
            a.click();
            URL.revokeObjectURL(a.href);
        } else {
            writeFile(file, data, error => {
                if (error) throw error;
            });
        }
    }

}