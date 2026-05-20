import * as tf from '@tensorflow/tfjs'

import { perplexityFromLoss, type NextTokenExample } from '../08-training-loop-cpu/index.js'

export type TrainableMiniTransformerOptions = {
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly seed?: number
}

export type TrainableMiniTransformer = {
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly tokenEmbeddings: tf.Variable
    readonly positionEmbeddings: tf.Variable
    readonly queryWeights: tf.Variable
    readonly keyWeights: tf.Variable
    readonly valueWeights: tf.Variable
    readonly attentionOutputWeights: tf.Variable
    readonly feedForwardInputWeights: tf.Variable
    readonly feedForwardInputBias: tf.Variable
    readonly feedForwardOutputWeights: tf.Variable
    readonly feedForwardOutputBias: tf.Variable
    readonly outputWeights: tf.Variable
    readonly outputBias: tf.Variable
}

export type MiniTransformerTrainingOptions = {
    readonly epochs: number
    readonly learningRate: number
}

export type MiniTransformerEpochMetrics = {
    readonly epoch: number
    readonly averageLoss: number
    readonly perplexity: number
}

export type MiniTransformerTrainingHistory = {
    readonly initialLoss: number
    readonly finalLoss: number
    readonly epochs: readonly MiniTransformerEpochMetrics[]
}

export type MiniTransformerGenerationOptions = {
    readonly maxNewTokens: number
}

export type MiniTransformerGenerationStep = {
    readonly step: number
    readonly contextTokenIds: readonly number[]
    readonly selectedTokenId: number
    readonly tokenIdsAfterPrediction: readonly number[]
}

export type MiniTransformerTokenGenerationResult = {
    readonly initialTokenIds: readonly number[]
    readonly generatedTokenIds: readonly number[]
    readonly tokenIds: readonly number[]
    readonly steps: readonly MiniTransformerGenerationStep[]
}

export type MiniTransformerTextGenerationTokenizer = {
    readonly encode: (text: string) => number[]
    readonly decode: (tokenIds: number[]) => string
}

export type MiniTransformerTextGenerationResult = MiniTransformerTokenGenerationResult & {
    readonly prompt: string
    readonly generatedText: string
    readonly text: string
}

const defaultSeed = 14
const initializationStdDev = 0.02
const futureMaskValue = -1_000_000_000

export function createTrainableMiniTransformer(
    options: TrainableMiniTransformerOptions,
): TrainableMiniTransformer {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')
    validatePositiveInteger(options.contextLength, 'contextLength')
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')
    validatePositiveInteger(options.feedForwardDimension, 'feedForwardDimension')

    const seed = options.seed ?? defaultSeed

    return {
        attentionOutputWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 5,
            ),
            true,
            'attentionOutputWeights',
        ),
        contextLength: options.contextLength,
        embeddingDimension: options.embeddingDimension,
        feedForwardDimension: options.feedForwardDimension,
        feedForwardInputBias: tf.variable(
            tf.zeros([options.feedForwardDimension]),
            true,
            'feedForwardInputBias',
        ),
        feedForwardInputWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.feedForwardDimension],
                seed + 6,
            ),
            true,
            'feedForwardInputWeights',
        ),
        feedForwardOutputBias: tf.variable(
            tf.zeros([options.embeddingDimension]),
            true,
            'feedForwardOutputBias',
        ),
        feedForwardOutputWeights: tf.variable(
            createSmallRandomTensor(
                [options.feedForwardDimension, options.embeddingDimension],
                seed + 7,
            ),
            true,
            'feedForwardOutputWeights',
        ),
        keyWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 3,
            ),
            true,
            'keyWeights',
        ),
        outputBias: tf.variable(tf.zeros([options.vocabularySize]), true, 'outputBias'),
        outputWeights: tf.variable(
            createSmallRandomTensor([options.embeddingDimension, options.vocabularySize], seed + 8),
            true,
            'outputWeights',
        ),
        positionEmbeddings: tf.variable(
            createSmallRandomTensor([options.contextLength, options.embeddingDimension], seed + 1),
            true,
            'positionEmbeddings',
        ),
        queryWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 2,
            ),
            true,
            'queryWeights',
        ),
        tokenEmbeddings: tf.variable(
            createSmallRandomTensor([options.vocabularySize, options.embeddingDimension], seed),
            true,
            'tokenEmbeddings',
        ),
        valueWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 4,
            ),
            true,
            'valueWeights',
        ),
        vocabularySize: options.vocabularySize,
    }
}

export function predictMiniTransformerLogits(
    model: TrainableMiniTransformer,
    inputTokenIds: readonly number[],
): tf.Tensor1D {
    validateContext(model, inputTokenIds)

    return tf.tidy(() => {
        const inputTensor = tf.tensor2d([[...inputTokenIds]], [1, model.contextLength], 'int32')
        const logits = computeBatchLogits(model, inputTensor)

        return logits.squeeze([0])
    })
}

export function predictMiniTransformerProbabilities(
    model: TrainableMiniTransformer,
    inputTokenIds: readonly number[],
): readonly number[] {
    const probabilities = tf.tidy(() =>
        tf.softmax(predictMiniTransformerLogits(model, inputTokenIds)),
    )
    const values = Array.from(probabilities.dataSync())

    probabilities.dispose()

    return values
}

export function predictMiniTransformerNextToken(
    model: TrainableMiniTransformer,
    inputTokenIds: readonly number[],
): number {
    const probabilities = predictMiniTransformerProbabilities(model, inputTokenIds)
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

export function generateMiniTransformerTokenIds(
    model: TrainableMiniTransformer,
    initialTokenIds: readonly number[],
    options: MiniTransformerGenerationOptions,
): MiniTransformerTokenGenerationResult {
    validatePositiveInteger(options.maxNewTokens, 'maxNewTokens')

    if (initialTokenIds.length < model.contextLength) {
        throw new Error(
            `Le prompt doit contenir au moins ${String(
                model.contextLength,
            )} tokens. Nombre reçu: ${String(initialTokenIds.length)}.`,
        )
    }

    const tokenIds = [...initialTokenIds]
    const generatedTokenIds: number[] = []
    const steps: MiniTransformerGenerationStep[] = []

    for (let step = 1; step <= options.maxNewTokens; step++) {
        const contextTokenIds = tokenIds.slice(-model.contextLength)
        const selectedTokenId = predictMiniTransformerNextToken(model, contextTokenIds)

        tokenIds.push(selectedTokenId)
        generatedTokenIds.push(selectedTokenId)
        steps.push({
            contextTokenIds,
            selectedTokenId,
            step,
            tokenIdsAfterPrediction: [...tokenIds],
        })
    }

    return {
        generatedTokenIds,
        initialTokenIds: [...initialTokenIds],
        steps,
        tokenIds,
    }
}

export function generateMiniTransformerText(
    model: TrainableMiniTransformer,
    tokenizer: MiniTransformerTextGenerationTokenizer,
    prompt: string,
    options: MiniTransformerGenerationOptions,
): MiniTransformerTextGenerationResult {
    const initialTokenIds = tokenizer.encode(prompt)
    const tokenResult = generateMiniTransformerTokenIds(model, initialTokenIds, options)
    const generatedText = tokenizer.decode([...tokenResult.generatedTokenIds])
    const text = tokenizer.decode([...tokenResult.tokenIds])

    return {
        ...tokenResult,
        generatedText,
        prompt,
        text,
    }
}

export function computeMiniTransformerAverageLoss(
    model: TrainableMiniTransformer,
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

export function trainMiniTransformer(
    model: TrainableMiniTransformer,
    examples: readonly NextTokenExample[],
    options: MiniTransformerTrainingOptions,
): MiniTransformerTrainingHistory {
    validateTrainingOptions(options)
    validateExamples(model, examples)

    const tensors = createExampleTensors(model, examples)
    const optimizer = tf.train.adam(options.learningRate)
    const initialLoss = computeLossValue(model, tensors.inputTokenIds, tensors.targetTokenIds)
    const epochMetrics: MiniTransformerEpochMetrics[] = []

    try {
        for (let epoch = 1; epoch <= options.epochs; epoch++) {
            const cost = optimizer.minimize(
                () => computeLossTensor(model, tensors.inputTokenIds, tensors.targetTokenIds),
                true,
                getTrainableVariables(model),
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

export function disposeTrainableMiniTransformer(model: TrainableMiniTransformer): void {
    for (const variable of getTrainableVariables(model)) {
        variable.dispose()
    }
}

function computeBatchLogits(
    model: TrainableMiniTransformer,
    inputTokenIds: tf.Tensor2D,
): tf.Tensor2D {
    return tf.tidy(() => {
        const contextualVectors = computeTransformerContextualVectors(model, inputTokenIds)
        const lastPosition = contextualVectors
            .slice([0, model.contextLength - 1, 0], [-1, 1, -1])
            .squeeze([1])

        return lastPosition.matMul(model.outputWeights).add(model.outputBias)
    })
}

function computeTransformerContextualVectors(
    model: TrainableMiniTransformer,
    inputTokenIds: tf.Tensor2D,
): tf.Tensor3D {
    return tf.tidy(() => {
        const tokenEmbeddings = tf.gather(model.tokenEmbeddings, inputTokenIds) as tf.Tensor3D
        const inputVectors = asTensor3D(tokenEmbeddings.add(model.positionEmbeddings.expandDims(0)))
        const batchSize = inputTokenIds.shape[0]

        const queries = projectSequence(inputVectors, model.queryWeights, model.embeddingDimension)
        const keys = projectSequence(inputVectors, model.keyWeights, model.embeddingDimension)
        const values = projectSequence(inputVectors, model.valueWeights, model.embeddingDimension)

        const scores = tf
            .matMul(queries, keys, false, true)
            .div(Math.sqrt(model.embeddingDimension))
            .add(createCausalMask(model.contextLength).expandDims(0))
        const attentionWeights = tf.softmax(scores, -1)
        const attentionValues = asTensor3D(tf.matMul(attentionWeights, values))
        const attentionCorrection = projectSequence(
            attentionValues,
            model.attentionOutputWeights,
            model.embeddingDimension,
        )
        const attentionResidual = asTensor3D(inputVectors.add(attentionCorrection))

        const feedForwardInput = attentionResidual
            .reshape([batchSize * model.contextLength, model.embeddingDimension])
            .matMul(model.feedForwardInputWeights)
            .add(model.feedForwardInputBias)
            .relu()
        const feedForwardCorrection = feedForwardInput
            .matMul(model.feedForwardOutputWeights)
            .add(model.feedForwardOutputBias)
            .reshape([batchSize, model.contextLength, model.embeddingDimension])

        return asTensor3D(attentionResidual.add(feedForwardCorrection))
    })
}

function asTensor3D(tensor: tf.Tensor): tf.Tensor3D {
    if (tensor.shape.length !== 3) {
        throw new Error(`Tensor 3D attendu. Shape reçue: [${tensor.shape.join(', ')}].`)
    }

    return tensor as tf.Tensor3D
}

function projectSequence(
    sequence: tf.Tensor3D,
    weights: tf.Variable,
    outputDimension: number,
): tf.Tensor3D {
    const [batchSize, sequenceLength, inputDimension] = sequence.shape

    return sequence
        .reshape([batchSize * sequenceLength, inputDimension])
        .matMul(weights)
        .reshape([batchSize, sequenceLength, outputDimension])
}

function computeLossTensor(
    model: TrainableMiniTransformer,
    inputTokenIds: tf.Tensor2D,
    targetTokenIds: tf.Tensor1D,
): tf.Scalar {
    return tf.tidy(() => {
        const logits = computeBatchLogits(model, inputTokenIds)
        const logProbabilities = tf.logSoftmax(logits)
        const targetsOneHot = tf.oneHot(targetTokenIds, model.vocabularySize)

        return targetsOneHot.mul(logProbabilities).sum(1).neg().mean()
    })
}

function computeLossValue(
    model: TrainableMiniTransformer,
    inputTokenIds: tf.Tensor2D,
    targetTokenIds: tf.Tensor1D,
): number {
    const loss = computeLossTensor(model, inputTokenIds, targetTokenIds)
    const value = readScalarTensor(loss, 'loss')

    loss.dispose()

    return value
}

function createExampleTensors(
    model: TrainableMiniTransformer,
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

function createCausalMask(contextLength: number): tf.Tensor2D {
    const values = Array.from({ length: contextLength }, (_, rowIndex) =>
        Array.from({ length: contextLength }, (_unused, columnIndex) =>
            columnIndex > rowIndex ? futureMaskValue : 0,
        ),
    )

    return tf.tensor2d(values, [contextLength, contextLength])
}

function createSmallRandomTensor(shape: readonly [number, number], seed: number): tf.Tensor2D {
    return tf.randomNormal([shape[0], shape[1]], 0, initializationStdDev, 'float32', seed)
}

function getTrainableVariables(model: TrainableMiniTransformer): tf.Variable[] {
    return [
        model.tokenEmbeddings,
        model.positionEmbeddings,
        model.queryWeights,
        model.keyWeights,
        model.valueWeights,
        model.attentionOutputWeights,
        model.feedForwardInputWeights,
        model.feedForwardInputBias,
        model.feedForwardOutputWeights,
        model.feedForwardOutputBias,
        model.outputWeights,
        model.outputBias,
    ]
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

function validateContext(model: TrainableMiniTransformer, inputTokenIds: readonly number[]): void {
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

function validateExamples(
    model: TrainableMiniTransformer,
    examples: readonly NextTokenExample[],
): void {
    if (examples.length === 0) {
        throw new Error('Le mini Transformer attend au moins un exemple.')
    }

    for (const example of examples) {
        validateContext(model, example.inputTokenIds)
        validateTokenId(example.targetTokenId, model.vocabularySize, 'targetTokenId')
    }
}

function validateModelShape(model: TrainableMiniTransformer): void {
    validatePositiveInteger(model.vocabularySize, 'model.vocabularySize')
    validatePositiveInteger(model.contextLength, 'model.contextLength')
    validatePositiveInteger(model.embeddingDimension, 'model.embeddingDimension')
    validatePositiveInteger(model.feedForwardDimension, 'model.feedForwardDimension')

    assertShape(model.tokenEmbeddings.shape, [model.vocabularySize, model.embeddingDimension])
    assertShape(model.positionEmbeddings.shape, [model.contextLength, model.embeddingDimension])
    assertShape(model.queryWeights.shape, [model.embeddingDimension, model.embeddingDimension])
    assertShape(model.keyWeights.shape, [model.embeddingDimension, model.embeddingDimension])
    assertShape(model.valueWeights.shape, [model.embeddingDimension, model.embeddingDimension])
    assertShape(model.attentionOutputWeights.shape, [
        model.embeddingDimension,
        model.embeddingDimension,
    ])
    assertShape(model.feedForwardInputWeights.shape, [
        model.embeddingDimension,
        model.feedForwardDimension,
    ])
    assertShape(model.feedForwardInputBias.shape, [model.feedForwardDimension])
    assertShape(model.feedForwardOutputWeights.shape, [
        model.feedForwardDimension,
        model.embeddingDimension,
    ])
    assertShape(model.feedForwardOutputBias.shape, [model.embeddingDimension])
    assertShape(model.outputWeights.shape, [model.embeddingDimension, model.vocabularySize])
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

function validateTrainingOptions(options: MiniTransformerTrainingOptions): void {
    validatePositiveInteger(options.epochs, 'epochs')
    validatePositiveNumber(options.learningRate, 'learningRate')
}
