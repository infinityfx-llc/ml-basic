import { Classifier } from '../classifiers/classifier';

export function fromFile(path: string): Promise<Classifier>;

export function fromFile(file: File): Promise<Classifier>;