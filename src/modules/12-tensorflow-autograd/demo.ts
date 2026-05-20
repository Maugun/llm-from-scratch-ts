import * as tf from '@tensorflow/tfjs'

import {
    createScalarRegressionModel,
    disposeScalarRegressionModel,
    meanSquaredError,
    predict,
    trainScalarRegressionModel,
    type ScalarRegressionExample,
    type ScalarRegressionModel,
} from './index.js'

const examples: readonly ScalarRegressionExample[] = [
    { input: 0, target: 1 },
    { input: 1, target: 3 },
    { input: 2, target: 5 },
    { input: 3, target: 7 },
]

const epochs = 80
const learningRate = 0.05

const model = createScalarRegressionModel({
    initialBias: 0,
    initialWeight: 0,
})

console.info('Module 12 - TensorFlow.js / Autograd')
console.info('')
console.info('But du module:')
console.info(
    '  Comprendre comment TensorFlow.js entraîne des paramètres sans écrire les gradients à la main.',
)
console.info('')
console.info("Pourquoi cet exemple n'est pas un LLM ?")
console.info('  On utilise une formule volontairement simple: y = weight * x + bias.')
console.info("  Elle sert de loupe sur l'autograd avant de revenir aux tokens au module 13.")
console.info('')
console.info('Pipeline:')
console.info('1. Créer des exemples x -> y')
console.info('2. Créer des tf.Variable entraînables: weight et bias')
console.info('3. Calculer une prédiction avec des tf.Tensor')
console.info('4. Mesurer la loss')
console.info('5. Laisser TensorFlow.js calculer les gradients')
console.info("6. Laisser l'optimizer mettre à jour les variables")
console.info('')
console.info('Prédictions attendues:')

for (const example of examples) {
    console.info(`  x = ${String(example.input)} -> attendu ${String(example.target)}`)
}

console.info('')
console.info('Relation attendue: y = 2x + 1')
console.info(`Epochs: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('')
console.info('Avant entraînement:')
printModelState(model)
console.info(`  loss: ${computeDemoLoss(model, examples).toFixed(4)}`)
console.info('')
console.info('Prédictions avant entraînement:')
printPredictions(model, [0, 1, 2, 3, 4])

const history = trainScalarRegressionModel(model, examples, {
    epochs,
    learningRate,
})

console.info('')
console.info('Pendant l’entraînement:')

for (const metrics of history.epochs) {
    if (metrics.epoch === 1 || metrics.epoch % 10 === 0 || metrics.epoch === epochs) {
        console.info(
            `  epoch ${String(metrics.epoch).padStart(2, ' ')} | loss ${metrics.loss.toFixed(
                4,
            )} | weight ${metrics.weight.toFixed(3)} | bias ${metrics.bias.toFixed(3)}`,
        )
    }
}

console.info('')
console.info('Après entraînement:')
printModelState(model)
console.info(`  loss finale: ${history.finalLoss.toFixed(4)}`)
console.info('')
console.info('Prédictions après entraînement:')
printPredictions(model, [0, 1, 2, 3, 4])

console.info('')
console.info('À retenir:')
console.info('  weight et bias sont des variables entraînables.')
console.info('  La loss indique si la prédiction est mauvaise.')
console.info("  L'autograd calcule automatiquement comment corriger les variables.")
console.info('  Le module 13 appliquera cette mécanique à une prédiction next-token.')

disposeScalarRegressionModel(model)

function computeDemoLoss(
    scalarModel: ScalarRegressionModel,
    scalarExamples: readonly ScalarRegressionExample[],
): number {
    const inputs = tf.tensor1d(scalarExamples.map((example) => example.input))
    const targets = tf.tensor1d(scalarExamples.map((example) => example.target))
    const loss = tf.tidy(() => {
        const predictions = predict(scalarModel, inputs)

        return meanSquaredError(predictions, targets)
    })
    const value = loss.dataSync()[0]

    inputs.dispose()
    targets.dispose()
    loss.dispose()

    if (value === undefined) {
        throw new Error('La loss de démo ne contient aucune valeur.')
    }

    return value
}

function predictNumber(scalarModel: ScalarRegressionModel, input: number): number {
    const inputTensor = tf.scalar(input)
    const prediction = predict(scalarModel, inputTensor)
    const value = prediction.dataSync()[0]

    inputTensor.dispose()
    prediction.dispose()

    if (value === undefined) {
        throw new Error('La prédiction de démo ne contient aucune valeur.')
    }

    return value
}

function printPredictions(scalarModel: ScalarRegressionModel, inputs: readonly number[]): void {
    for (const input of inputs) {
        console.info(
            `  x = ${String(input)} -> prédiction ${predictNumber(scalarModel, input).toFixed(3)}`,
        )
    }
}

function printModelState(scalarModel: ScalarRegressionModel): void {
    const weight = scalarModel.weight.dataSync()[0]
    const bias = scalarModel.bias.dataSync()[0]

    if (weight === undefined || bias === undefined) {
        throw new Error('Le modèle de démo ne contient pas weight ou bias.')
    }

    console.info(`  weight: ${weight.toFixed(3)}`)
    console.info(`  bias:   ${bias.toFixed(3)}`)
}
