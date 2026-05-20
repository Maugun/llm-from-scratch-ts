import * as tf from '@tensorflow/tfjs'
import { afterEach, describe, expect, it } from 'vitest'

import {
    createScalarRegressionModel,
    disposeScalarRegressionModel,
    meanSquaredError,
    predict,
    trainScalarRegressionModel,
    type ScalarRegressionModel,
} from '../../src/modules/12-tensorflow-autograd/index.js'

const modelsToDispose: ScalarRegressionModel[] = []

afterEach(() => {
    for (const model of modelsToDispose.splice(0)) {
        disposeScalarRegressionModel(model)
    }
})

describe('createScalarRegressionModel', () => {
    it('crée un modèle avec weight et bias', () => {
        const model = trackModel(
            createScalarRegressionModel({
                initialBias: 1,
                initialWeight: 2,
            }),
        )

        expect(readScalar(model.weight)).toBe(2)
        expect(readScalar(model.bias)).toBe(1)
    })
})

describe('predict', () => {
    it('calcule weight * input + bias avec des valeurs fixées', () => {
        const model = trackModel(
            createScalarRegressionModel({
                initialBias: 1,
                initialWeight: 2,
            }),
        )
        const input = tf.tensor1d([0, 1, 2])
        const prediction = predict(model, input)

        expect(Array.from(prediction.dataSync())).toEqual([1, 3, 5])

        input.dispose()
        prediction.dispose()
    })
})

describe('meanSquaredError', () => {
    it('retourne 0 quand prédiction et cible sont identiques', () => {
        const predictions = tf.tensor1d([1, 2, 3])
        const targets = tf.tensor1d([1, 2, 3])
        const loss = meanSquaredError(predictions, targets)

        expect(readScalar(loss)).toBe(0)

        predictions.dispose()
        targets.dispose()
        loss.dispose()
    })

    it('retourne une valeur positive quand la prédiction est fausse', () => {
        const predictions = tf.tensor1d([1, 2, 3])
        const targets = tf.tensor1d([2, 2, 2])
        const loss = meanSquaredError(predictions, targets)

        expect(readScalar(loss)).toBeGreaterThan(0)

        predictions.dispose()
        targets.dispose()
        loss.dispose()
    })
})

describe('trainScalarRegressionModel', () => {
    it('réduit la loss sur un dataset linéaire simple', () => {
        const model = trackModel(createScalarRegressionModel())
        const examples = [
            { input: 0, target: 1 },
            { input: 1, target: 3 },
            { input: 2, target: 5 },
            { input: 3, target: 7 },
        ]

        const history = trainScalarRegressionModel(model, examples, {
            epochs: 120,
            learningRate: 0.05,
        })

        expect(history.finalLoss).toBeLessThan(history.initialLoss)
    })

    it('apprend des valeurs proches de weight = 2 et bias = 1', () => {
        const model = trackModel(createScalarRegressionModel())

        trainScalarRegressionModel(
            model,
            [
                { input: 0, target: 1 },
                { input: 1, target: 3 },
                { input: 2, target: 5 },
                { input: 3, target: 7 },
            ],
            {
                epochs: 250,
                learningRate: 0.05,
            },
        )

        expect(readScalar(model.weight)).toBeCloseTo(2, 1)
        expect(readScalar(model.bias)).toBeCloseTo(1, 1)
    })

    it('rejette epochs <= 0', () => {
        const model = trackModel(createScalarRegressionModel())

        expect(() =>
            trainScalarRegressionModel(model, [{ input: 0, target: 1 }], {
                epochs: 0,
                learningRate: 0.1,
            }),
        ).toThrow('epochs doit être un entier strictement positif.')
    })

    it('rejette learningRate <= 0', () => {
        const model = trackModel(createScalarRegressionModel())

        expect(() =>
            trainScalarRegressionModel(model, [{ input: 0, target: 1 }], {
                epochs: 1,
                learningRate: 0,
            }),
        ).toThrow('learningRate doit être un nombre fini strictement positif.')
    })

    it('rejette un dataset vide', () => {
        const model = trackModel(createScalarRegressionModel())

        expect(() =>
            trainScalarRegressionModel(model, [], {
                epochs: 1,
                learningRate: 0.1,
            }),
        ).toThrow('trainScalarRegressionModel attend au moins un exemple.')
    })
})

describe('disposeScalarRegressionModel', () => {
    it('peut être appelé sans erreur', () => {
        const model = createScalarRegressionModel()

        expect(() => {
            disposeScalarRegressionModel(model)
        }).not.toThrow()
    })
})

function trackModel(model: ScalarRegressionModel): ScalarRegressionModel {
    modelsToDispose.push(model)

    return model
}

function readScalar(tensor: tf.Tensor): number {
    const value = tensor.dataSync()[0]

    if (value === undefined) {
        throw new Error('Le tenseur de test ne contient aucune valeur.')
    }

    return value
}
