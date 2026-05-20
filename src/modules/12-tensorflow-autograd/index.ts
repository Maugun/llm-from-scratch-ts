import * as tf from '@tensorflow/tfjs'

export type ScalarRegressionExample = {
    readonly input: number
    readonly target: number
}

export type ScalarRegressionOptions = {
    /**
     * Valeur initiale de weight.
     *
     * Dans un vrai modèle, les paramètres commencent souvent avec de petites valeurs
     * aléatoires. Ici, on laisse le choix pour rendre les tests et la démo déterministes.
     */
    readonly initialWeight?: number

    /**
     * Valeur initiale de bias.
     */
    readonly initialBias?: number
}

export type ScalarRegressionModel = {
    /**
     * Paramètre entraînable qui multiplie l'entrée x.
     *
     * C'est une tf.Variable: TensorFlow.js peut donc la modifier pendant l'entraînement.
     */
    readonly weight: tf.Variable

    /**
     * Paramètre entraînable ajouté après la multiplication.
     */
    readonly bias: tf.Variable
}

export type TensorTrainingOptions = {
    readonly epochs: number
    readonly learningRate: number
}

export type TensorEpochMetrics = {
    readonly epoch: number
    readonly loss: number
    readonly weight: number
    readonly bias: number
}

export type TensorTrainingHistory = {
    readonly initialLoss: number
    readonly finalLoss: number
    readonly epochs: readonly TensorEpochMetrics[]
}

const defaultInitialWeight = 0
const defaultInitialBias = 0

/**
 * Crée un mini modèle y = weight * x + bias.
 *
 * Pourquoi une régression aussi simple dans un projet LLM ?
 * Parce qu'elle isole la mécanique que l'on veut comprendre:
 * variables entraînables -> prédiction -> loss -> gradients automatiques -> update.
 */
export function createScalarRegressionModel(
    options: ScalarRegressionOptions = {},
): ScalarRegressionModel {
    const initialWeight = options.initialWeight ?? defaultInitialWeight
    const initialBias = options.initialBias ?? defaultInitialBias

    validateFiniteNumber(initialWeight, 'initialWeight')
    validateFiniteNumber(initialBias, 'initialBias')

    return {
        bias: tf.variable(tf.scalar(initialBias), true, 'bias'),
        weight: tf.variable(tf.scalar(initialWeight), true, 'weight'),
    }
}

/**
 * Calcule y = weight * x + bias.
 *
 * input est un tenseur, pas un simple number[]. Cette différence est centrale:
 * TensorFlow.js peut suivre les opérations faites sur des tenseurs et construire le chemin
 * nécessaire au calcul automatique des gradients.
 */
export function predict(model: ScalarRegressionModel, input: tf.Tensor): tf.Tensor {
    return tf.tidy(() => input.mul(model.weight).add(model.bias))
}

/**
 * Mean squared error: moyenne des erreurs au carré.
 *
 * Si prediction = target, l'erreur vaut 0. Plus la prédiction s'éloigne de la cible,
 * plus la loss augmente. Le carré évite que les erreurs positives et négatives s'annulent.
 */
export function meanSquaredError(predictions: tf.Tensor, targets: tf.Tensor): tf.Scalar {
    return tf.tidy(() => predictions.sub(targets).square().mean())
}

/**
 * Entraîne le mini modèle avec l'autograd TensorFlow.js.
 *
 * On ne code pas la dérivée de y = weight * x + bias à la main. L'optimizer observe la loss,
 * demande les gradients à TensorFlow.js, puis corrige weight et bias pour réduire cette loss.
 */
export function trainScalarRegressionModel(
    model: ScalarRegressionModel,
    examples: readonly ScalarRegressionExample[],
    options: TensorTrainingOptions,
): TensorTrainingHistory {
    validateTrainingInputs(examples, options)

    const inputs = tf.tensor1d(examples.map((example) => example.input))
    const targets = tf.tensor1d(examples.map((example) => example.target))
    const optimizer = tf.train.sgd(options.learningRate)
    const epochMetrics: TensorEpochMetrics[] = []
    const initialLoss = computeLossValue(model, inputs, targets)

    try {
        for (let epoch = 1; epoch <= options.epochs; epoch++) {
            const cost = optimizer.minimize(
                () => {
                    const predictions = predict(model, inputs)

                    return meanSquaredError(predictions, targets)
                },
                true,
                [model.weight, model.bias],
            )

            cost?.dispose()

            const loss = computeLossValue(model, inputs, targets)

            epochMetrics.push({
                bias: readScalarVariable(model.bias, 'bias'),
                epoch,
                loss,
                weight: readScalarVariable(model.weight, 'weight'),
            })
        }

        return {
            epochs: epochMetrics,
            finalLoss: computeLossValue(model, inputs, targets),
            initialLoss,
        }
    } finally {
        inputs.dispose()
        targets.dispose()
    }
}

export function disposeScalarRegressionModel(model: ScalarRegressionModel): void {
    model.weight.dispose()
    model.bias.dispose()
}

function computeLossValue(
    model: ScalarRegressionModel,
    inputs: tf.Tensor,
    targets: tf.Tensor,
): number {
    const loss = tf.tidy(() => {
        const predictions = predict(model, inputs)

        return meanSquaredError(predictions, targets)
    })
    const value = loss.dataSync()[0]

    loss.dispose()

    if (value === undefined) {
        throw new Error('La loss TensorFlow.js ne contient aucune valeur.')
    }

    return value
}

function readScalarVariable(variable: tf.Variable, name: string): number {
    const value = variable.dataSync()[0]

    if (value === undefined) {
        throw new Error(`${name} ne contient aucune valeur scalaire.`)
    }

    return value
}

function validateFiniteNumber(value: number, name: string): void {
    if (!Number.isFinite(value)) {
        throw new Error(`${name} doit être un nombre fini. Valeur reçue: ${String(value)}.`)
    }
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
            `${name} doit être un entier strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validatePositiveNumber(value: number, name: string): void {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(
            `${name} doit être un nombre fini strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validateTrainingInputs(
    examples: readonly ScalarRegressionExample[],
    options: TensorTrainingOptions,
): void {
    validatePositiveInteger(options.epochs, 'epochs')
    validatePositiveNumber(options.learningRate, 'learningRate')

    if (examples.length === 0) {
        throw new Error('trainScalarRegressionModel attend au moins un exemple.')
    }

    for (const [index, example] of examples.entries()) {
        validateFiniteNumber(example.input, `examples[${String(index)}].input`)
        validateFiniteNumber(example.target, `examples[${String(index)}].target`)
    }
}
