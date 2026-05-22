import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import * as tf from '@tensorflow/tfjs'

import { perplexityFromLoss } from '../08-training-loop-cpu/index.js'
import {
    selectNextToken,
    type SamplingSelectionOptions,
    type SamplingStrategy,
} from '../11-sampling-strategies/index.js'
import {
    disposeTensorNextTokenBatch,
    estimateNextTokenExampleCount,
    getBatchCount,
    iterateNextTokenBatches,
    nextTokenBatchToTensors,
    type LongCorpusPipeline,
    type NextTokenBatch,
    type NextTokenBatchOptions,
} from '../17-long-corpus-pipeline/index.js'
export {
    loadBpeTokenizer,
    saveBpeTokenizer,
    trainBpeTokenizer,
    encodeWithBpe,
    decodeWithBpe,
} from './bpe-tokenizer.js'
export type {
    BpeMerge,
    BpeTokenizer,
    BpeTokenizerTrainingOptions,
    BpeTokenizerTrainingProgress,
} from './bpe-tokenizer.js'
import { loadBpeTokenizer, saveBpeTokenizer, type BpeTokenizer } from './bpe-tokenizer.js'

export type FinalTinyLlmOptions = {
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly layerCount: number
    readonly headCount: number
    readonly seed?: number
}

export type FinalTinyLlmLayer = {
    readonly layerNorm1Scale: tf.Variable
    readonly layerNorm1Bias: tf.Variable
    readonly queryWeights: tf.Variable
    readonly keyWeights: tf.Variable
    readonly valueWeights: tf.Variable
    readonly attentionOutputWeights: tf.Variable
    readonly layerNorm2Scale: tf.Variable
    readonly layerNorm2Bias: tf.Variable
    readonly feedForwardInputWeights: tf.Variable
    readonly feedForwardInputBias: tf.Variable
    readonly feedForwardOutputWeights: tf.Variable
    readonly feedForwardOutputBias: tf.Variable
}

export type FinalTinyLlm = {
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly layerCount: number
    readonly headCount: number
    readonly headDimension: number
    readonly tokenEmbeddings: tf.Variable
    readonly positionEmbeddings: tf.Variable
    readonly layers: readonly FinalTinyLlmLayer[]
    readonly finalLayerNormScale: tf.Variable
    readonly finalLayerNormBias: tf.Variable
    readonly outputWeights: tf.Variable
    readonly outputBias: tf.Variable
}

export type FinalTinyLlmTrainingBatchOrder = 'sequential' | 'shuffled'

export type FinalTinyLlmTrainingOptions = {
    readonly epochs: number
    readonly learningRate: number
    readonly batchSize?: number
    readonly maxTrainBatchesPerEpoch?: number
    readonly maxValidationBatches?: number
    readonly batchOrder?: FinalTinyLlmTrainingBatchOrder
    readonly shuffleSeed?: number
    readonly saveBestEpochOnly?: boolean
    readonly onProgress?: (progress: FinalTinyLlmTrainingProgress) => void
}

export type FinalTinyLlmTrainingProgress = {
    readonly epoch: number
    readonly epochs: number
    readonly trainedBatches: number
    readonly totalBatchesInEpoch: number
    readonly progressRatio: number
    readonly latestBatchLoss: number
    readonly elapsedMs: number
}

export type FinalTinyLlmEpochMetrics = {
    readonly epoch: number
    readonly trainLoss: number
    readonly trainPerplexity: number
    readonly validationLoss: number
    readonly validationPerplexity: number
    readonly trainedBatches: number
}

export type FinalTinyLlmTrainingHistory = {
    readonly initialValidationLoss: number
    readonly initialValidationPerplexity: number
    readonly finalValidationLoss: number
    readonly finalValidationPerplexity: number
    readonly bestEpoch: number
    readonly bestValidationLoss: number
    readonly bestValidationPerplexity: number
    readonly restoredBestEpochWeights: boolean
    readonly epochs: readonly FinalTinyLlmEpochMetrics[]
}

export type FinalTinyLlmEvaluationOptions = {
    readonly batchSize: number
    readonly maxBatches?: number
}

export type FinalTinyLlmEvaluationMetrics = {
    readonly averageLoss: number
    readonly perplexity: number
    readonly evaluatedBatches: number
    readonly evaluatedExamples: number
}

export type FinalTinyLlmGenerationOptions = {
    readonly maxNewTokens: number
    readonly strategy: SamplingStrategy
    readonly temperature?: number
    readonly topK?: number
    readonly seed?: number
    readonly onProgress?: (step: FinalTinyLlmGenerationStep) => void
}

export type FinalTinyLlmGenerationStep = {
    readonly step: number
    readonly strategy: SamplingStrategy
    readonly contextTokenIds: readonly number[]
    readonly selectedTokenId: number
    readonly selectedTokenProbability: number
    readonly tokenIdsAfterPrediction: readonly number[]
}

export type FinalTinyLlmTextGenerationResult = {
    readonly prompt: string
    readonly generatedText: string
    readonly text: string
    readonly initialTokenIds: readonly number[]
    readonly generatedTokenIds: readonly number[]
    readonly tokenIds: readonly number[]
    readonly steps: readonly FinalTinyLlmGenerationStep[]
}

export type FinalTinyLlmChatMessage = {
    readonly role: 'user' | 'assistant'
    readonly content: string
}

export type FinalTinyLlmCheckpointMetadata = {
    readonly version: 1
    readonly createdAt: string
    readonly modelOptions: FinalTinyLlmOptions
    readonly tokenizerFileName: string
    readonly extra?: Record<string, unknown>
    readonly variables: readonly FinalTinyLlmCheckpointVariable[]
}

export type FinalTinyLlmCheckpointVariable = {
    readonly name: string
    readonly fileName: string
    readonly shape: readonly number[]
    readonly dtype: 'float32'
}

export type FinalTinyLlmCheckpointLoadProgress = {
    readonly phase: 'metadata' | 'tokenizer' | 'model' | 'variables' | 'done'
    readonly loadedVariables: number
    readonly totalVariables: number
    readonly currentVariableName: string | undefined
    readonly elapsedMs: number
}

export type LoadFinalTinyLlmCheckpointOptions = {
    readonly onProgress?: (progress: FinalTinyLlmCheckpointLoadProgress) => void
}

export type SaveFinalTinyLlmCheckpointMetadata = {
    readonly extra?: Record<string, unknown>
}

type FinalTinyLlmVariableSnapshot = {
    readonly name: string
    readonly shape: readonly number[]
    readonly values: Float32Array
}

const defaultSeed = 19
const initializationStdDev = 0.02
const futureMaskValue = -1_000_000_000
const layerNormEpsilon = 1e-5
const checkpointVersion = 1
const checkpointMetadataFileName = 'metadata.json'
const tokenizerFileName = 'tokenizer.json'
let modelInstanceCounter = 0

export function createFinalTinyLlm(options: FinalTinyLlmOptions): FinalTinyLlm {
    validateModelOptions(options)

    const seed = options.seed ?? defaultSeed
    const variablePrefix = `finalTinyLlm_${String(modelInstanceCounter++)}`
    const headDimension = options.embeddingDimension / options.headCount
    const layers = Array.from({ length: options.layerCount }, (_unused, layerIndex) =>
        createLayer(options, seed + 100 * (layerIndex + 1), layerIndex, variablePrefix),
    )

    return {
        contextLength: options.contextLength,
        embeddingDimension: options.embeddingDimension,
        feedForwardDimension: options.feedForwardDimension,
        finalLayerNormBias: tf.variable(
            tf.zeros([options.embeddingDimension]),
            true,
            `${variablePrefix}_finalLayerNormBias`,
        ),
        finalLayerNormScale: tf.variable(
            tf.ones([options.embeddingDimension]),
            true,
            `${variablePrefix}_finalLayerNormScale`,
        ),
        headCount: options.headCount,
        headDimension,
        layerCount: options.layerCount,
        layers,
        outputBias: tf.variable(
            tf.zeros([options.vocabularySize]),
            true,
            `${variablePrefix}_outputBias`,
        ),
        outputWeights: tf.variable(
            createSmallRandomTensor([options.embeddingDimension, options.vocabularySize], seed + 2),
            true,
            `${variablePrefix}_outputWeights`,
        ),
        positionEmbeddings: tf.variable(
            createSmallRandomTensor([options.contextLength, options.embeddingDimension], seed + 1),
            true,
            `${variablePrefix}_positionEmbeddings`,
        ),
        tokenEmbeddings: tf.variable(
            createSmallRandomTensor([options.vocabularySize, options.embeddingDimension], seed),
            true,
            `${variablePrefix}_tokenEmbeddings`,
        ),
        vocabularySize: options.vocabularySize,
    }
}

export function predictFinalTinyLlmProbabilities(
    model: FinalTinyLlm,
    inputTokenIds: readonly number[],
): readonly number[] {
    const probabilities = tf.tidy(() => tf.softmax(predictFinalTinyLlmLogits(model, inputTokenIds)))
    const values = Array.from(probabilities.dataSync())

    probabilities.dispose()

    return values
}

export function predictFinalTinyLlmNextToken(
    model: FinalTinyLlm,
    inputTokenIds: readonly number[],
    samplingOptions: SamplingSelectionOptions = { strategy: 'greedy' },
): number {
    const probabilities = predictFinalTinyLlmProbabilities(model, inputTokenIds)

    return selectNextToken(probabilities, samplingOptions)
}

export function generateFinalTinyLlmText(
    model: FinalTinyLlm,
    tokenizer: BpeTokenizer,
    prompt: string,
    options: FinalTinyLlmGenerationOptions,
): FinalTinyLlmTextGenerationResult {
    validatePositiveInteger(options.maxNewTokens, 'maxNewTokens')

    const random = createDeterministicRandom(options.seed ?? 1234)
    const initialTokenIds = tokenizer.encode(prompt)

    if (initialTokenIds.length < model.contextLength) {
        throw new Error(
            `Le prompt doit contenir au moins ${String(
                model.contextLength,
            )} tokens BPE. Nombre reçu: ${String(initialTokenIds.length)}.`,
        )
    }

    const tokenIds = [...initialTokenIds]
    const generatedTokenIds: number[] = []
    const steps: FinalTinyLlmGenerationStep[] = []

    for (let step = 1; step <= options.maxNewTokens; step++) {
        const contextTokenIds = tokenIds.slice(-model.contextLength)
        const probabilities = predictFinalTinyLlmProbabilities(model, contextTokenIds)
        const selectedTokenId = selectNextToken(
            probabilities,
            createSamplingSelectionOptions(options, random),
        )
        const selectedTokenProbability = readNumberAt(probabilities, selectedTokenId)

        tokenIds.push(selectedTokenId)
        generatedTokenIds.push(selectedTokenId)
        const generationStep = {
            contextTokenIds,
            selectedTokenId,
            selectedTokenProbability,
            step,
            strategy: options.strategy,
            tokenIdsAfterPrediction: [...tokenIds],
        }

        steps.push(generationStep)
        options.onProgress?.(generationStep)
    }

    return {
        generatedText: tokenizer.decode(generatedTokenIds),
        generatedTokenIds,
        initialTokenIds,
        prompt,
        steps,
        text: tokenizer.decode(tokenIds),
        tokenIds,
    }
}

export function chatWithFinalTinyLlm(
    model: FinalTinyLlm,
    tokenizer: BpeTokenizer,
    messages: readonly FinalTinyLlmChatMessage[],
    options: FinalTinyLlmGenerationOptions,
): FinalTinyLlmTextGenerationResult {
    const prompt = formatChatPrompt(tokenizer, messages)

    return generateFinalTinyLlmText(model, tokenizer, prompt, options)
}

export function evaluateFinalTinyLlm(
    model: FinalTinyLlm,
    tokenIds: readonly number[],
    options: FinalTinyLlmEvaluationOptions,
): FinalTinyLlmEvaluationMetrics {
    validatePositiveInteger(options.batchSize, 'batchSize')

    if (options.maxBatches !== undefined) {
        validatePositiveInteger(options.maxBatches, 'maxBatches')
    }

    let totalLoss = 0
    let evaluatedExamples = 0
    let evaluatedBatches = 0

    for (const batch of iterateNextTokenBatches(
        tokenIds,
        createBatchOptions(model, options.batchSize),
    )) {
        if (options.maxBatches !== undefined && evaluatedBatches >= options.maxBatches) {
            break
        }

        const tensorBatch = nextTokenBatchToTensors(batch)
        const loss = computeLossValue(model, tensorBatch.inputTokenIds, tensorBatch.targetTokenIds)
        const batchExampleCount = batch.targetTokenIds.length

        totalLoss += loss * batchExampleCount
        evaluatedExamples += batchExampleCount
        evaluatedBatches++
        disposeTensorNextTokenBatch(tensorBatch)
    }

    if (evaluatedExamples === 0) {
        throw new Error('Aucun exemple disponible pour évaluer le LLM final.')
    }

    const averageLoss = totalLoss / evaluatedExamples

    return {
        averageLoss,
        evaluatedBatches,
        evaluatedExamples,
        perplexity: perplexityFromLoss(averageLoss),
    }
}

export function trainFinalTinyLlm(
    model: FinalTinyLlm,
    pipeline: LongCorpusPipeline,
    options: FinalTinyLlmTrainingOptions,
): FinalTinyLlmTrainingHistory {
    validateTrainingOptions(options)
    validatePipelineCompatibility(model, pipeline)

    const batchSize = options.batchSize ?? pipeline.batchSize
    const maxValidationBatches = options.maxValidationBatches ?? 5
    const maxTrainBatches =
        options.maxTrainBatchesPerEpoch === undefined
            ? pipeline.trainBatchCount
            : Math.min(options.maxTrainBatchesPerEpoch, pipeline.trainBatchCount)
    const optimizer = tf.train.adam(options.learningRate)
    const initialValidation = evaluateFinalTinyLlm(model, pipeline.validationTokenIds, {
        batchSize,
        maxBatches: maxValidationBatches,
    })
    const epochMetrics: FinalTinyLlmEpochMetrics[] = []
    const trainingStartedAt = Date.now()
    let bestEpoch = 0
    let bestValidationLoss = initialValidation.averageLoss
    let bestValidationPerplexity = initialValidation.perplexity
    let bestWeights = options.saveBestEpochOnly === true ? cloneVariableSnapshot(model) : undefined

    for (let epoch = 1; epoch <= options.epochs; epoch++) {
        let totalTrainLoss = 0
        let trainedExamples = 0
        let trainedBatches = 0

        for (const batch of iterateTrainingBatches(
            pipeline.trainTokenIds,
            createBatchOptions(model, batchSize),
            options.batchOrder ?? 'shuffled',
            (options.shuffleSeed ?? defaultSeed) + epoch,
        )) {
            if (trainedBatches >= maxTrainBatches) {
                break
            }

            const tensorBatch = nextTokenBatchToTensors(batch)
            const cost = optimizer.minimize(
                () =>
                    computeLossTensor(model, tensorBatch.inputTokenIds, tensorBatch.targetTokenIds),
                true,
                getTrainableVariables(model),
            )
            const loss = cost === null ? Number.NaN : readScalarTensor(cost, 'loss')
            const batchExampleCount = batch.targetTokenIds.length

            cost?.dispose()
            totalTrainLoss += loss * batchExampleCount
            trainedExamples += batchExampleCount
            trainedBatches++
            options.onProgress?.({
                elapsedMs: Date.now() - trainingStartedAt,
                epoch,
                epochs: options.epochs,
                latestBatchLoss: loss,
                progressRatio: trainedBatches / maxTrainBatches,
                totalBatchesInEpoch: maxTrainBatches,
                trainedBatches,
            })
            disposeTensorNextTokenBatch(tensorBatch)
        }

        if (trainedExamples === 0) {
            throw new Error('Aucun batch d’entraînement disponible.')
        }

        const trainLoss = totalTrainLoss / trainedExamples
        const validation = evaluateFinalTinyLlm(model, pipeline.validationTokenIds, {
            batchSize,
            maxBatches: maxValidationBatches,
        })

        if (validation.averageLoss < bestValidationLoss) {
            bestEpoch = epoch
            bestValidationLoss = validation.averageLoss
            bestValidationPerplexity = validation.perplexity
            bestWeights =
                options.saveBestEpochOnly === true ? cloneVariableSnapshot(model) : bestWeights
        }

        epochMetrics.push({
            epoch,
            trainLoss,
            trainPerplexity: perplexityFromLoss(trainLoss),
            trainedBatches,
            validationLoss: validation.averageLoss,
            validationPerplexity: validation.perplexity,
        })
    }

    if (options.saveBestEpochOnly === true && bestWeights !== undefined) {
        restoreVariableSnapshot(model, bestWeights)
    }

    const finalValidation = evaluateFinalTinyLlm(model, pipeline.validationTokenIds, {
        batchSize,
        maxBatches: maxValidationBatches,
    })

    return {
        epochs: epochMetrics,
        bestEpoch,
        bestValidationLoss,
        bestValidationPerplexity,
        finalValidationLoss: finalValidation.averageLoss,
        finalValidationPerplexity: finalValidation.perplexity,
        initialValidationLoss: initialValidation.averageLoss,
        initialValidationPerplexity: initialValidation.perplexity,
        restoredBestEpochWeights: options.saveBestEpochOnly === true,
    }
}

export async function saveFinalTinyLlmCheckpoint(
    model: FinalTinyLlm,
    tokenizer: BpeTokenizer,
    directoryPath: string,
    metadata: SaveFinalTinyLlmCheckpointMetadata = {},
): Promise<FinalTinyLlmCheckpointMetadata> {
    await mkdir(directoryPath, { recursive: true })
    await saveBpeTokenizer(tokenizer, join(directoryPath, tokenizerFileName))

    const variables = getNamedVariables(model)
    const checkpointVariables: FinalTinyLlmCheckpointVariable[] = []

    for (const variable of variables) {
        const fileName = `${variable.name}.bin`
        const values = await variable.tensor.data()
        const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength)

        await writeFile(join(directoryPath, fileName), bytes)
        checkpointVariables.push({
            dtype: 'float32',
            fileName,
            name: variable.name,
            shape: variable.tensor.shape,
        })
    }

    const checkpointMetadata: FinalTinyLlmCheckpointMetadata = {
        createdAt: new Date().toISOString(),
        modelOptions: getModelOptions(model),
        tokenizerFileName,
        variables: checkpointVariables,
        version: checkpointVersion,
        ...(metadata.extra === undefined ? {} : { extra: metadata.extra }),
    }

    await writeFile(
        join(directoryPath, checkpointMetadataFileName),
        `${JSON.stringify(checkpointMetadata, null, 2)}\n`,
        'utf8',
    )

    return checkpointMetadata
}

export async function loadFinalTinyLlmCheckpoint(
    directoryPath: string,
    options: LoadFinalTinyLlmCheckpointOptions = {},
): Promise<{ readonly model: FinalTinyLlm; readonly tokenizer: BpeTokenizer }> {
    const startedAt = Date.now()
    const notifyProgress = (
        phase: FinalTinyLlmCheckpointLoadProgress['phase'],
        loadedVariables = 0,
        totalVariables = 0,
        currentVariableName?: string,
    ): void => {
        options.onProgress?.({
            currentVariableName,
            elapsedMs: Date.now() - startedAt,
            loadedVariables,
            phase,
            totalVariables,
        })
    }

    notifyProgress('metadata')
    const metadata = await readCheckpointMetadata(directoryPath)
    notifyProgress('tokenizer', 0, metadata.variables.length)
    const tokenizer = await loadBpeTokenizer(join(directoryPath, metadata.tokenizerFileName))
    notifyProgress('model', 0, metadata.variables.length)
    const model = createFinalTinyLlm(metadata.modelOptions)
    const variableByName = new Map(
        getNamedVariables(model).map((variable) => [variable.name, variable]),
    )
    const loadedVariableNames = new Set<string>()

    try {
        for (const [variableIndex, variableMetadata] of metadata.variables.entries()) {
            const variable = variableByName.get(variableMetadata.name)

            if (variable === undefined) {
                throw new Error(`Variable inconnue dans le checkpoint: ${variableMetadata.name}.`)
            }

            notifyProgress(
                'variables',
                variableIndex,
                metadata.variables.length,
                variableMetadata.name,
            )
            assertShape(variable.tensor.shape, variableMetadata.shape)

            const rawBuffer = await readFile(join(directoryPath, variableMetadata.fileName))
            const values = readFloat32Values(rawBuffer, variableMetadata.name)
            const tensor = tf.tensor(values, [...variableMetadata.shape], 'float32')

            variable.tensor.assign(tensor)
            tensor.dispose()
            loadedVariableNames.add(variableMetadata.name)
            notifyProgress(
                'variables',
                variableIndex + 1,
                metadata.variables.length,
                variableMetadata.name,
            )
        }

        for (const variableName of variableByName.keys()) {
            if (!loadedVariableNames.has(variableName)) {
                throw new Error(`Variable manquante dans le checkpoint: ${variableName}.`)
            }
        }

        notifyProgress('done', metadata.variables.length, metadata.variables.length)

        return { model, tokenizer }
    } catch (error) {
        disposeFinalTinyLlm(model)
        throw error
    }
}

export function disposeFinalTinyLlm(model: FinalTinyLlm): void {
    for (const variable of getTrainableVariables(model)) {
        variable.dispose()
    }
}

function createLayer(
    options: FinalTinyLlmOptions,
    seed: number,
    layerIndex: number,
    variablePrefix: string,
): FinalTinyLlmLayer {
    return {
        attentionOutputWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 3,
            ),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_attentionOutputWeights`,
        ),
        feedForwardInputBias: tf.variable(
            tf.zeros([options.feedForwardDimension]),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_feedForwardInputBias`,
        ),
        feedForwardInputWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.feedForwardDimension],
                seed + 4,
            ),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_feedForwardInputWeights`,
        ),
        feedForwardOutputBias: tf.variable(
            tf.zeros([options.embeddingDimension]),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_feedForwardOutputBias`,
        ),
        feedForwardOutputWeights: tf.variable(
            createSmallRandomTensor(
                [options.feedForwardDimension, options.embeddingDimension],
                seed + 5,
            ),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_feedForwardOutputWeights`,
        ),
        keyWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 1,
            ),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_keyWeights`,
        ),
        layerNorm1Bias: tf.variable(
            tf.zeros([options.embeddingDimension]),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_layerNorm1Bias`,
        ),
        layerNorm1Scale: tf.variable(
            tf.ones([options.embeddingDimension]),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_layerNorm1Scale`,
        ),
        layerNorm2Bias: tf.variable(
            tf.zeros([options.embeddingDimension]),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_layerNorm2Bias`,
        ),
        layerNorm2Scale: tf.variable(
            tf.ones([options.embeddingDimension]),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_layerNorm2Scale`,
        ),
        queryWeights: tf.variable(
            createSmallRandomTensor([options.embeddingDimension, options.embeddingDimension], seed),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_queryWeights`,
        ),
        valueWeights: tf.variable(
            createSmallRandomTensor(
                [options.embeddingDimension, options.embeddingDimension],
                seed + 2,
            ),
            true,
            `${variablePrefix}_layer${String(layerIndex)}_valueWeights`,
        ),
    }
}

function predictFinalTinyLlmLogits(
    model: FinalTinyLlm,
    inputTokenIds: readonly number[],
): tf.Tensor1D {
    validateContext(model, inputTokenIds)

    return tf.tidy(() => {
        const inputTensor = tf.tensor2d([[...inputTokenIds]], [1, model.contextLength], 'int32')
        const logits = computeBatchLogits(model, inputTensor)

        return logits.squeeze([0])
    })
}

function computeBatchLogits(model: FinalTinyLlm, inputTokenIds: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
        const contextualVectors = computeContextualVectors(model, inputTokenIds)
        const normalized = layerNormalize(
            contextualVectors,
            model.finalLayerNormScale,
            model.finalLayerNormBias,
        )
        const lastVector = normalized
            .slice([0, model.contextLength - 1, 0], [-1, 1, -1])
            .squeeze([1])

        return lastVector.matMul(model.outputWeights).add(model.outputBias)
    })
}

function computeContextualVectors(model: FinalTinyLlm, inputTokenIds: tf.Tensor2D): tf.Tensor3D {
    return tf.tidy(() => {
        const tokenEmbeddings = tf.gather(model.tokenEmbeddings, inputTokenIds) as tf.Tensor3D
        const positionEmbeddings = model.positionEmbeddings.expandDims(0)
        let sequence = asTensor3D(tokenEmbeddings.add(positionEmbeddings))

        for (const layer of model.layers) {
            sequence = applyTransformerLayer(model, layer, sequence)
        }

        return sequence
    })
}

function applyTransformerLayer(
    model: FinalTinyLlm,
    layer: FinalTinyLlmLayer,
    inputVectors: tf.Tensor3D,
): tf.Tensor3D {
    return tf.tidy(() => {
        const normalizedForAttention = layerNormalize(
            inputVectors,
            layer.layerNorm1Scale,
            layer.layerNorm1Bias,
        )
        const attentionCorrection = multiHeadCausalAttention(model, layer, normalizedForAttention)
        const attentionResidual = asTensor3D(inputVectors.add(attentionCorrection))
        const normalizedForFeedForward = layerNormalize(
            attentionResidual,
            layer.layerNorm2Scale,
            layer.layerNorm2Bias,
        )
        const [batchSize] = inputVectors.shape
        const feedForwardInput = normalizedForFeedForward
            .reshape([batchSize * model.contextLength, model.embeddingDimension])
            .matMul(layer.feedForwardInputWeights)
            .add(layer.feedForwardInputBias)
            .relu()
        const feedForwardCorrection = feedForwardInput
            .matMul(layer.feedForwardOutputWeights)
            .add(layer.feedForwardOutputBias)
            .reshape([batchSize, model.contextLength, model.embeddingDimension])

        return asTensor3D(attentionResidual.add(feedForwardCorrection))
    })
}

function multiHeadCausalAttention(
    model: FinalTinyLlm,
    layer: FinalTinyLlmLayer,
    inputVectors: tf.Tensor3D,
): tf.Tensor3D {
    return tf.tidy(() => {
        const queries = splitHeads(projectSequence(inputVectors, layer.queryWeights, model))
        const keys = splitHeads(projectSequence(inputVectors, layer.keyWeights, model))
        const values = splitHeads(projectSequence(inputVectors, layer.valueWeights, model))
        const scores = tf
            .matMul(queries, keys, false, true)
            .div(Math.sqrt(model.headDimension))
            .add(createCausalMask(model.contextLength).expandDims(0).expandDims(0))
        const attentionWeights = tf.softmax(scores, -1)
        const attentionValues = tf.matMul(attentionWeights, values)
        const mergedHeads = mergeHeads(asTensor4D(attentionValues), model)

        return projectSequence(mergedHeads, layer.attentionOutputWeights, model)

        function splitHeads(sequence: tf.Tensor3D): tf.Tensor4D {
            const [batchSize] = sequence.shape

            return sequence
                .reshape([batchSize, model.contextLength, model.headCount, model.headDimension])
                .transpose([0, 2, 1, 3])
        }
    })
}

function mergeHeads(values: tf.Tensor4D, model: FinalTinyLlm): tf.Tensor3D {
    const [batchSize] = values.shape

    return values
        .transpose([0, 2, 1, 3])
        .reshape([batchSize, model.contextLength, model.embeddingDimension])
}

function layerNormalize(
    inputVectors: tf.Tensor3D,
    scale: tf.Variable,
    bias: tf.Variable,
): tf.Tensor3D {
    return tf.tidy(() => {
        const mean = inputVectors.mean(-1, true)
        const variance = inputVectors.sub(mean).square().mean(-1, true)
        const normalized = inputVectors.sub(mean).div(variance.add(layerNormEpsilon).sqrt())

        return asTensor3D(normalized.mul(scale).add(bias))
    })
}

function computeLossTensor(
    model: FinalTinyLlm,
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
    model: FinalTinyLlm,
    inputTokenIds: tf.Tensor2D,
    targetTokenIds: tf.Tensor1D,
): number {
    const loss = computeLossTensor(model, inputTokenIds, targetTokenIds)
    const value = readScalarTensor(loss, 'loss')

    loss.dispose()

    return value
}

function createBatchOptions(model: FinalTinyLlm, batchSize: number): NextTokenBatchOptions {
    return {
        batchSize,
        contextLength: model.contextLength,
    }
}

function* iterateTrainingBatches(
    tokenIds: readonly number[],
    options: NextTokenBatchOptions,
    batchOrder: FinalTinyLlmTrainingBatchOrder,
    seed: number,
): Generator<NextTokenBatch> {
    if (batchOrder === 'sequential') {
        yield* iterateNextTokenBatches(tokenIds, options)
        return
    }

    const exampleCount = estimateNextTokenExampleCount(tokenIds.length, options.contextLength)
    const batchCount = getBatchCount(exampleCount, options.batchSize)
    const batchIndices = shuffleNumbers(
        Array.from({ length: batchCount }, (_unused, batchIndex) => batchIndex),
        seed,
    )

    for (const [shuffledBatchIndex, originalBatchIndex] of batchIndices.entries()) {
        const batch = createNextTokenBatchFromStartIndex(
            tokenIds,
            options,
            originalBatchIndex * options.batchSize,
            shuffledBatchIndex,
            exampleCount,
        )

        if (batch !== undefined) {
            yield batch
        }
    }
}

function createNextTokenBatchFromStartIndex(
    tokenIds: readonly number[],
    options: NextTokenBatchOptions,
    startExampleIndex: number,
    batchIndex: number,
    exampleCount: number,
): NextTokenBatch | undefined {
    if (startExampleIndex >= exampleCount) {
        return undefined
    }

    const inputTokenIds: number[][] = []
    const targetTokenIds: number[] = []
    const endExampleIndex = Math.min(startExampleIndex + options.batchSize, exampleCount)

    for (let exampleIndex = startExampleIndex; exampleIndex < endExampleIndex; exampleIndex++) {
        const targetIndex = exampleIndex + options.contextLength
        const targetTokenId = tokenIds[targetIndex]

        if (targetTokenId === undefined) {
            throw new Error(`Token introuvable à l’index ${String(targetIndex)}.`)
        }

        inputTokenIds.push(tokenIds.slice(exampleIndex, targetIndex))
        targetTokenIds.push(targetTokenId)
    }

    return {
        batchIndex,
        inputTokenIds,
        startExampleIndex,
        targetTokenIds,
    }
}

function shuffleNumbers(values: readonly number[], seed: number): readonly number[] {
    const shuffled = [...values]
    const random = createDeterministicRandom(seed)

    for (let index = shuffled.length - 1; index > 0; index--) {
        const swapIndex = Math.floor(random() * (index + 1))
        const currentValue = shuffled[index]
        const swapValue = shuffled[swapIndex]

        if (currentValue === undefined || swapValue === undefined) {
            throw new Error('Index de shuffle invalide.')
        }

        shuffled[index] = swapValue
        shuffled[swapIndex] = currentValue
    }

    return shuffled
}

function createCausalMask(contextLength: number): tf.Tensor2D {
    const values = Array.from({ length: contextLength }, (_unused, rowIndex) =>
        Array.from({ length: contextLength }, (_unusedColumn, columnIndex) =>
            columnIndex > rowIndex ? futureMaskValue : 0,
        ),
    )

    return tf.tensor2d(values, [contextLength, contextLength])
}

function createSmallRandomTensor(shape: readonly [number, number], seed: number): tf.Tensor2D {
    return tf.randomNormal([shape[0], shape[1]], 0, initializationStdDev, 'float32', seed)
}

function createDeterministicRandom(seed: number): () => number {
    let state = seed >>> 0

    return () => {
        state = (state * 1664525 + 1013904223) % 4294967296

        return state / 4294967296
    }
}

function createSamplingSelectionOptions(
    options: FinalTinyLlmGenerationOptions,
    random: () => number,
): SamplingSelectionOptions {
    return {
        random,
        strategy: options.strategy,
        ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
        ...(options.topK === undefined ? {} : { topK: options.topK }),
    }
}

function formatChatPrompt(
    tokenizer: BpeTokenizer,
    messages: readonly FinalTinyLlmChatMessage[],
): string {
    const formattedMessages = messages.map((message) =>
        message.role === 'user'
            ? `Utilisateur: ${message.content}`
            : `Assistant: ${message.content}`,
    )
    const labeledPrompt = `${formattedMessages.join('\n')}\nAssistant:`

    if (canEncode(tokenizer, labeledPrompt)) {
        return labeledPrompt
    }

    return messages.map((message) => message.content).join('\n')
}

function canEncode(tokenizer: BpeTokenizer, text: string): boolean {
    try {
        tokenizer.encode(text)

        return true
    } catch {
        return false
    }
}

function getTrainableVariables(model: FinalTinyLlm): tf.Variable[] {
    return getNamedVariables(model).map((variable) => variable.tensor)
}

function cloneVariableSnapshot(model: FinalTinyLlm): readonly FinalTinyLlmVariableSnapshot[] {
    return getNamedVariables(model).map((variable) => ({
        name: variable.name,
        shape: variable.tensor.shape,
        values: new Float32Array(variable.tensor.dataSync()),
    }))
}

function restoreVariableSnapshot(
    model: FinalTinyLlm,
    snapshots: readonly FinalTinyLlmVariableSnapshot[],
): void {
    const variableByName = new Map(
        getNamedVariables(model).map((variable) => [variable.name, variable.tensor]),
    )

    for (const snapshot of snapshots) {
        const variable = variableByName.get(snapshot.name)

        if (variable === undefined) {
            throw new Error(`Snapshot invalide: variable inconnue ${snapshot.name}.`)
        }

        assertShape(variable.shape, snapshot.shape)

        const tensor = tf.tensor(snapshot.values, [...snapshot.shape], 'float32')

        variable.assign(tensor)
        tensor.dispose()
    }
}

function getNamedVariables(
    model: FinalTinyLlm,
): readonly { readonly name: string; readonly tensor: tf.Variable }[] {
    return [
        { name: 'tokenEmbeddings', tensor: model.tokenEmbeddings },
        { name: 'positionEmbeddings', tensor: model.positionEmbeddings },
        ...model.layers.flatMap((layer, layerIndex) => [
            { name: `layers.${String(layerIndex)}.layerNorm1Scale`, tensor: layer.layerNorm1Scale },
            { name: `layers.${String(layerIndex)}.layerNorm1Bias`, tensor: layer.layerNorm1Bias },
            { name: `layers.${String(layerIndex)}.queryWeights`, tensor: layer.queryWeights },
            { name: `layers.${String(layerIndex)}.keyWeights`, tensor: layer.keyWeights },
            { name: `layers.${String(layerIndex)}.valueWeights`, tensor: layer.valueWeights },
            {
                name: `layers.${String(layerIndex)}.attentionOutputWeights`,
                tensor: layer.attentionOutputWeights,
            },
            { name: `layers.${String(layerIndex)}.layerNorm2Scale`, tensor: layer.layerNorm2Scale },
            { name: `layers.${String(layerIndex)}.layerNorm2Bias`, tensor: layer.layerNorm2Bias },
            {
                name: `layers.${String(layerIndex)}.feedForwardInputWeights`,
                tensor: layer.feedForwardInputWeights,
            },
            {
                name: `layers.${String(layerIndex)}.feedForwardInputBias`,
                tensor: layer.feedForwardInputBias,
            },
            {
                name: `layers.${String(layerIndex)}.feedForwardOutputWeights`,
                tensor: layer.feedForwardOutputWeights,
            },
            {
                name: `layers.${String(layerIndex)}.feedForwardOutputBias`,
                tensor: layer.feedForwardOutputBias,
            },
        ]),
        { name: 'finalLayerNormScale', tensor: model.finalLayerNormScale },
        { name: 'finalLayerNormBias', tensor: model.finalLayerNormBias },
        { name: 'outputWeights', tensor: model.outputWeights },
        { name: 'outputBias', tensor: model.outputBias },
    ]
}

function getModelOptions(model: FinalTinyLlm): FinalTinyLlmOptions {
    return {
        contextLength: model.contextLength,
        embeddingDimension: model.embeddingDimension,
        feedForwardDimension: model.feedForwardDimension,
        headCount: model.headCount,
        layerCount: model.layerCount,
        vocabularySize: model.vocabularySize,
    }
}

async function readCheckpointMetadata(
    directoryPath: string,
): Promise<FinalTinyLlmCheckpointMetadata> {
    const rawJson = await readFile(join(directoryPath, checkpointMetadataFileName), 'utf8')
    const parsedValue = JSON.parse(rawJson) as unknown

    return validateCheckpointMetadata(parsedValue)
}

function validateCheckpointMetadata(value: unknown): FinalTinyLlmCheckpointMetadata {
    if (!isRecord(value)) {
        throw new Error('Le checkpoint doit contenir un objet JSON.')
    }

    if (value.version !== checkpointVersion) {
        throw new Error(`Version de checkpoint invalide. Attendu: ${String(checkpointVersion)}.`)
    }

    assertString(value.createdAt, 'createdAt')
    assertString(value.tokenizerFileName, 'tokenizerFileName')

    if (!isRecord(value.modelOptions)) {
        throw new Error('modelOptions doit être un objet.')
    }

    const modelOptions = value.modelOptions
    const normalizedOptions: FinalTinyLlmOptions = {
        contextLength: readPositiveInteger(
            modelOptions.contextLength,
            'modelOptions.contextLength',
        ),
        embeddingDimension: readPositiveInteger(
            modelOptions.embeddingDimension,
            'modelOptions.embeddingDimension',
        ),
        feedForwardDimension: readPositiveInteger(
            modelOptions.feedForwardDimension,
            'modelOptions.feedForwardDimension',
        ),
        headCount: readPositiveInteger(modelOptions.headCount, 'modelOptions.headCount'),
        layerCount: readPositiveInteger(modelOptions.layerCount, 'modelOptions.layerCount'),
        vocabularySize: readPositiveInteger(
            modelOptions.vocabularySize,
            'modelOptions.vocabularySize',
        ),
    }

    validateModelOptions(normalizedOptions)

    if (!Array.isArray(value.variables)) {
        throw new Error('variables doit être un tableau.')
    }

    return {
        createdAt: value.createdAt,
        modelOptions: normalizedOptions,
        tokenizerFileName: value.tokenizerFileName,
        variables: value.variables.map(validateCheckpointVariable),
        version: checkpointVersion,
        ...(isRecord(value.extra) ? { extra: value.extra } : {}),
    }
}

function validateCheckpointVariable(value: unknown): FinalTinyLlmCheckpointVariable {
    if (!isRecord(value)) {
        throw new Error('Chaque variable de checkpoint doit être un objet.')
    }

    assertString(value.name, 'variable.name')
    assertString(value.fileName, 'variable.fileName')

    if (
        value.fileName.includes('/') ||
        value.fileName.includes('\\') ||
        value.fileName.includes('..')
    ) {
        throw new Error('variable.fileName doit être un nom de fichier local simple.')
    }

    if (value.dtype !== 'float32') {
        throw new Error('Seul le dtype float32 est supporté pour les checkpoints.')
    }

    if (
        !Array.isArray(value.shape) ||
        value.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)
    ) {
        throw new Error('variable.shape doit contenir uniquement des entiers positifs.')
    }

    return {
        dtype: 'float32',
        fileName: value.fileName,
        name: value.name,
        shape: value.shape,
    }
}

function asTensor3D(tensor: tf.Tensor): tf.Tensor3D {
    if (tensor.shape.length !== 3) {
        throw new Error(`Tensor 3D attendu. Shape reçue: [${tensor.shape.join(', ')}].`)
    }

    return tensor as tf.Tensor3D
}

function asTensor4D(tensor: tf.Tensor): tf.Tensor4D {
    if (tensor.shape.length !== 4) {
        throw new Error(`Tensor 4D attendu. Shape reçue: [${tensor.shape.join(', ')}].`)
    }

    return tensor as tf.Tensor4D
}

function projectSequence(
    sequence: tf.Tensor3D,
    weights: tf.Variable,
    model: FinalTinyLlm,
): tf.Tensor3D {
    const [batchSize, sequenceLength, inputDimension] = sequence.shape

    return sequence
        .reshape([batchSize * sequenceLength, inputDimension])
        .matMul(weights)
        .reshape([batchSize, sequenceLength, model.embeddingDimension])
}

function readFloat32Values(rawBuffer: Buffer, variableName: string): Float32Array {
    if (rawBuffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
        throw new Error(`Poids binaires invalides pour ${variableName}: taille non multiple de 4.`)
    }

    const arrayBuffer = rawBuffer.buffer.slice(
        rawBuffer.byteOffset,
        rawBuffer.byteOffset + rawBuffer.byteLength,
    )

    return new Float32Array(arrayBuffer)
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

function readPositiveInteger(value: unknown, name: string): number {
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
        throw new Error(`${name} doit être un entier strictement positif.`)
    }

    return value
}

function validatePipelineCompatibility(model: FinalTinyLlm, pipeline: LongCorpusPipeline): void {
    if (pipeline.vocabularySize !== model.vocabularySize) {
        throw new Error(
            `La taille du vocabulaire pipeline (${String(
                pipeline.vocabularySize,
            )}) doit correspondre au modèle (${String(model.vocabularySize)}).`,
        )
    }

    if (pipeline.contextLength !== model.contextLength) {
        throw new Error(
            `Le contextLength pipeline (${String(
                pipeline.contextLength,
            )}) doit correspondre au modèle (${String(model.contextLength)}).`,
        )
    }
}

function validateContext(model: FinalTinyLlm, inputTokenIds: readonly number[]): void {
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

function validateModelOptions(options: FinalTinyLlmOptions): void {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')
    validatePositiveInteger(options.contextLength, 'contextLength')
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')
    validatePositiveInteger(options.feedForwardDimension, 'feedForwardDimension')
    validatePositiveInteger(options.layerCount, 'layerCount')
    validatePositiveInteger(options.headCount, 'headCount')

    if (options.embeddingDimension % options.headCount !== 0) {
        throw new Error('embeddingDimension doit être divisible par headCount.')
    }
}

function validateModelShape(model: FinalTinyLlm): void {
    validateModelOptions(getModelOptions(model))
    assertShape(model.tokenEmbeddings.shape, [model.vocabularySize, model.embeddingDimension])
    assertShape(model.positionEmbeddings.shape, [model.contextLength, model.embeddingDimension])

    for (const layer of model.layers) {
        assertShape(layer.layerNorm1Scale.shape, [model.embeddingDimension])
        assertShape(layer.layerNorm1Bias.shape, [model.embeddingDimension])
        assertShape(layer.queryWeights.shape, [model.embeddingDimension, model.embeddingDimension])
        assertShape(layer.keyWeights.shape, [model.embeddingDimension, model.embeddingDimension])
        assertShape(layer.valueWeights.shape, [model.embeddingDimension, model.embeddingDimension])
        assertShape(layer.attentionOutputWeights.shape, [
            model.embeddingDimension,
            model.embeddingDimension,
        ])
        assertShape(layer.layerNorm2Scale.shape, [model.embeddingDimension])
        assertShape(layer.layerNorm2Bias.shape, [model.embeddingDimension])
        assertShape(layer.feedForwardInputWeights.shape, [
            model.embeddingDimension,
            model.feedForwardDimension,
        ])
        assertShape(layer.feedForwardInputBias.shape, [model.feedForwardDimension])
        assertShape(layer.feedForwardOutputWeights.shape, [
            model.feedForwardDimension,
            model.embeddingDimension,
        ])
        assertShape(layer.feedForwardOutputBias.shape, [model.embeddingDimension])
    }

    assertShape(model.finalLayerNormScale.shape, [model.embeddingDimension])
    assertShape(model.finalLayerNormBias.shape, [model.embeddingDimension])
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

function assertString(value: unknown, name: string): asserts value is string {
    if (typeof value !== 'string') {
        throw new Error(`${name} doit être une chaîne de caractères.`)
    }
}

function validateTrainingOptions(options: FinalTinyLlmTrainingOptions): void {
    validatePositiveInteger(options.epochs, 'epochs')
    validatePositiveNumber(options.learningRate, 'learningRate')

    if (options.batchSize !== undefined) {
        validatePositiveInteger(options.batchSize, 'batchSize')
    }

    if (options.maxTrainBatchesPerEpoch !== undefined) {
        validatePositiveInteger(options.maxTrainBatchesPerEpoch, 'maxTrainBatchesPerEpoch')
    }

    if (options.maxValidationBatches !== undefined) {
        validatePositiveInteger(options.maxValidationBatches, 'maxValidationBatches')
    }

    const batchOrder = options.batchOrder as string | undefined

    if (batchOrder !== undefined && batchOrder !== 'sequential' && batchOrder !== 'shuffled') {
        throw new Error('batchOrder doit valoir "sequential" ou "shuffled".')
    }

    if (options.shuffleSeed !== undefined) {
        validatePositiveInteger(options.shuffleSeed, 'shuffleSeed')
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

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null
}
