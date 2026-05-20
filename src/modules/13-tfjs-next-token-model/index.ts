import * as tf from '@tensorflow/tfjs'

import { perplexityFromLoss, type NextTokenExample } from '../08-training-loop-cpu/index.js'

export type TfjsNextTokenModelOptions = {
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly seed?: number
}

export type TfjsNextTokenModel = {
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number

    /**
     * Une ligne par token du vocabulaire.
     *
     * Contrairement au module 4, cette table est une tf.Variable: TensorFlow.js peut donc
     * l'ajuster pendant l'entraînement.
     */
    readonly tokenEmbeddings: tf.Variable

    /**
     * Une ligne par position dans la fenêtre de contexte.
     *
     * Elle permet au modèle de distinguer "le token vu en première position" du même token vu
     * plus tard dans le contexte.
     */
    readonly positionEmbeddings: tf.Variable

    /**
     * Projection finale vers les logits du vocabulaire.
     *
     * Le contexte aplati a contextLength * embeddingDimension valeurs. Cette matrice transforme
     * ce vecteur en un score brut par prochain token possible.
     */
    readonly outputWeights: tf.Variable
    readonly outputBias: tf.Variable
}

export type TfjsNextTokenTrainingOptions = {
    readonly epochs: number
    readonly learningRate: number
}

export type TfjsNextTokenEpochMetrics = {
    readonly epoch: number
    readonly averageLoss: number
    readonly perplexity: number
}

export type TfjsNextTokenTrainingHistory = {
    readonly initialLoss: number
    readonly finalLoss: number
    readonly epochs: readonly TfjsNextTokenEpochMetrics[]
}

const defaultSeed = 13
const initializationStdDev = 0.02

/**
 * Crée un mini modèle neuronal de prédiction next-token.
 *
 * Le modèle reste volontairement simple:
 * 1. transformer chaque token du contexte en embedding;
 * 2. ajouter un embedding de position;
 * 3. aplatir le contexte;
 * 4. projeter vers un logit par token du vocabulaire.
 */
export function createTfjsNextTokenModel(options: TfjsNextTokenModelOptions): TfjsNextTokenModel {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')
    validatePositiveInteger(options.contextLength, 'contextLength')
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')

    const seed = options.seed ?? defaultSeed
    const flattenedContextDimension = options.contextLength * options.embeddingDimension

    return {
        contextLength: options.contextLength,
        embeddingDimension: options.embeddingDimension,
        outputBias: tf.variable(tf.zeros([options.vocabularySize]), true, 'outputBias'),
        outputWeights: tf.variable(
            createSmallRandomTensor([flattenedContextDimension, options.vocabularySize], seed + 2),
            true,
            'outputWeights',
        ),
        positionEmbeddings: tf.variable(
            createSmallRandomTensor([options.contextLength, options.embeddingDimension], seed + 1),
            true,
            'positionEmbeddings',
        ),
        tokenEmbeddings: tf.variable(
            createSmallRandomTensor([options.vocabularySize, options.embeddingDimension], seed),
            true,
            'tokenEmbeddings',
        ),
        vocabularySize: options.vocabularySize,
    }
}

/**
 * Retourne les logits du prochain token pour un contexte.
 *
 * Un logit est un score brut. Il n'est pas encore une probabilité: il peut être négatif,
 * positif, grand ou petit. La softmax transformera ensuite ces scores en distribution.
 */
export function predictNextTokenLogits(
    model: TfjsNextTokenModel,
    inputTokenIds: readonly number[],
): tf.Tensor1D {
    validateContext(model, inputTokenIds)

    return tf.tidy(() => {
        const inputTensor = tf.tensor2d([[...inputTokenIds]], [1, model.contextLength], 'int32')
        const logits = computeBatchLogits(model, inputTensor)

        return logits.squeeze([0])
    })
}

export function predictNextTokenProbabilities(
    model: TfjsNextTokenModel,
    inputTokenIds: readonly number[],
): readonly number[] {
    const probabilities = tf.tidy(() => tf.softmax(predictNextTokenLogits(model, inputTokenIds)))
    const values = Array.from(probabilities.dataSync())

    probabilities.dispose()

    return values
}

export function predictMostLikelyNextToken(
    model: TfjsNextTokenModel,
    inputTokenIds: readonly number[],
): number {
    const probabilities = predictNextTokenProbabilities(model, inputTokenIds)
    let bestTokenId = 0
    let bestProbability = readNumberAt(probabilities, 0)

    for (let tokenId = 1; tokenId < probabilities.length; tokenId++) {
        const probability = readNumberAt(probabilities, tokenId)

        if (probability > bestProbability) {
            bestProbability = probability
            bestTokenId = tokenId
        }
    }

    return bestTokenId
}

export function computeTfjsNextTokenAverageLoss(
    model: TfjsNextTokenModel,
    examples: readonly NextTokenExample[],
): number {
    validateExamples(model, examples)

    const tensors = createExampleTensors(model, examples)
    const loss = computeLossTensor(model, tensors.inputTokenIds, tensors.targetTokenIds)
    const value = readScalarTensor(loss, 'loss')

    tensors.inputTokenIds.dispose()
    tensors.targetTokenIds.dispose()
    loss.dispose()

    return value
}

/**
 * Entraîne le modèle avec TensorFlow.js.
 *
 * La différence avec le module 9 est fondamentale: on ne code plus le gradient de chaque poids
 * à la main. TensorFlow.js suit les opérations sur les tenseurs, calcule les gradients et
 * l'optimizer corrige les variables entraînables.
 */
export function trainTfjsNextTokenModel(
    model: TfjsNextTokenModel,
    examples: readonly NextTokenExample[],
    options: TfjsNextTokenTrainingOptions,
): TfjsNextTokenTrainingHistory {
    validateTrainingOptions(options)
    validateExamples(model, examples)

    const tensors = createExampleTensors(model, examples)
    const optimizer = tf.train.adam(options.learningRate)
    const initialLoss = computeLossValue(model, tensors.inputTokenIds, tensors.targetTokenIds)
    const epochMetrics: TfjsNextTokenEpochMetrics[] = []

    try {
        for (let epoch = 1; epoch <= options.epochs; epoch++) {
            const cost = optimizer.minimize(
                () => computeLossTensor(model, tensors.inputTokenIds, tensors.targetTokenIds),
                true,
                [
                    model.tokenEmbeddings,
                    model.positionEmbeddings,
                    model.outputWeights,
                    model.outputBias,
                ],
            )

            cost?.dispose()

            const averageLoss = computeLossValue(
                model,
                tensors.inputTokenIds,
                tensors.targetTokenIds,
            )

            epochMetrics.push({
                averageLoss,
                epoch,
                perplexity: perplexityFromLoss(averageLoss),
            })
        }

        return {
            epochs: epochMetrics,
            finalLoss: computeLossValue(model, tensors.inputTokenIds, tensors.targetTokenIds),
            initialLoss,
        }
    } finally {
        tensors.inputTokenIds.dispose()
        tensors.targetTokenIds.dispose()
    }
}

export function disposeTfjsNextTokenModel(model: TfjsNextTokenModel): void {
    model.tokenEmbeddings.dispose()
    model.positionEmbeddings.dispose()
    model.outputWeights.dispose()
    model.outputBias.dispose()
}

function computeBatchLogits(model: TfjsNextTokenModel, inputTokenIds: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
        // inputTokenIds shape: [batchSize, contextLength]
        // tokenEmbeddings shape après gather: [batchSize, contextLength, embeddingDimension]
        const tokenEmbeddings = tf.gather(model.tokenEmbeddings, inputTokenIds) as tf.Tensor3D

        // positionEmbeddings shape: [contextLength, embeddingDimension]
        // expandDims(0) donne [1, contextLength, embeddingDimension], ce qui se diffuse sur
        // tout le batch. Chaque position reçoit donc son propre vecteur de position.
        const contextualEmbeddings = tokenEmbeddings.add(model.positionEmbeddings.expandDims(0))
        const flattenedContext = contextualEmbeddings.reshape([
            inputTokenIds.shape[0],
            model.contextLength * model.embeddingDimension,
        ])

        // logits shape: [batchSize, vocabularySize]
        return flattenedContext.matMul(model.outputWeights).add(model.outputBias)
    })
}

function computeLossTensor(
    model: TfjsNextTokenModel,
    inputTokenIds: tf.Tensor2D,
    targetTokenIds: tf.Tensor1D,
): tf.Scalar {
    return tf.tidy(() => {
        const logits = computeBatchLogits(model, inputTokenIds)
        const logProbabilities = tf.logSoftmax(logits)
        const targetsOneHot = tf.oneHot(targetTokenIds, model.vocabularySize)

        // Cross-entropy sparse expliquée sans magie:
        // on transforme la cible en one-hot, on garde la log-probabilité du bon token, puis
        // on prend l'opposé. Plus le modèle donne peu de probabilité à la bonne cible, plus la
        // loss est élevée.
        return targetsOneHot.mul(logProbabilities).sum(1).neg().mean()
    })
}

function computeLossValue(
    model: TfjsNextTokenModel,
    inputTokenIds: tf.Tensor2D,
    targetTokenIds: tf.Tensor1D,
): number {
    const loss = computeLossTensor(model, inputTokenIds, targetTokenIds)
    const value = readScalarTensor(loss, 'loss')

    loss.dispose()

    return value
}

function createExampleTensors(
    model: TfjsNextTokenModel,
    examples: readonly NextTokenExample[],
): {
    readonly inputTokenIds: tf.Tensor2D
    readonly targetTokenIds: tf.Tensor1D
} {
    return {
        inputTokenIds: tf.tensor2d(
            examples.map((example) => [...example.inputTokenIds]),
            [examples.length, model.contextLength],
            'int32',
        ),
        targetTokenIds: tf.tensor1d(
            examples.map((example) => example.targetTokenId),
            'int32',
        ),
    }
}

function createSmallRandomTensor(shape: readonly [number, number], seed: number): tf.Tensor2D {
    return tf.randomNormal([shape[0], shape[1]], 0, initializationStdDev, 'float32', seed)
}

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable à l'index ${String(index)}.`)
    }

    return value
}

function readScalarTensor(tensor: tf.Tensor, name: string): number {
    const value = tensor.dataSync()[0]

    if (value === undefined) {
        throw new Error(`${name} ne contient aucune valeur scalaire.`)
    }

    return value
}

function validateContext(model: TfjsNextTokenModel, inputTokenIds: readonly number[]): void {
    validateModelShape(model)

    if (inputTokenIds.length !== model.contextLength) {
        throw new Error(
            `Le contexte doit contenir ${String(
                model.contextLength,
            )} tokens. Nombre reçu: ${String(inputTokenIds.length)}.`,
        )
    }

    for (const [position, tokenId] of inputTokenIds.entries()) {
        validateTokenId(tokenId, model.vocabularySize, `inputTokenIds[${String(position)}]`)
    }
}

function validateExamples(model: TfjsNextTokenModel, examples: readonly NextTokenExample[]): void {
    if (examples.length === 0) {
        throw new Error('Le modèle TensorFlow.js attend au moins un exemple.')
    }

    for (const example of examples) {
        validateContext(model, example.inputTokenIds)
        validateTokenId(example.targetTokenId, model.vocabularySize, 'targetTokenId')
    }
}

function validateModelShape(model: TfjsNextTokenModel): void {
    validatePositiveInteger(model.vocabularySize, 'model.vocabularySize')
    validatePositiveInteger(model.contextLength, 'model.contextLength')
    validatePositiveInteger(model.embeddingDimension, 'model.embeddingDimension')

    assertShape(model.tokenEmbeddings.shape, [model.vocabularySize, model.embeddingDimension])
    assertShape(model.positionEmbeddings.shape, [model.contextLength, model.embeddingDimension])
    assertShape(model.outputWeights.shape, [
        model.contextLength * model.embeddingDimension,
        model.vocabularySize,
    ])
    assertShape(model.outputBias.shape, [model.vocabularySize])
}

function assertShape(actualShape: readonly number[], expectedShape: readonly number[]): void {
    if (
        actualShape.length !== expectedShape.length ||
        actualShape.some((dimension, index) => dimension !== expectedShape[index])
    ) {
        throw new Error(
            `Shape TensorFlow.js invalide. Attendu: [${expectedShape.join(
                ', ',
            )}], reçu: [${actualShape.join(', ')}].`,
        )
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

function validateTokenId(tokenId: number, vocabularySize: number, name: string): void {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
        throw new Error(
            `${name} doit être un entier entre 0 et ${String(
                vocabularySize - 1,
            )}. Valeur reçue: ${String(tokenId)}.`,
        )
    }
}

function validateTrainingOptions(options: TfjsNextTokenTrainingOptions): void {
    validatePositiveInteger(options.epochs, 'epochs')
    validatePositiveNumber(options.learningRate, 'learningRate')
}
