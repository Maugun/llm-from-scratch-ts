import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import * as tf from '@tensorflow/tfjs'

export type LongCorpusTokenizer = {
    readonly encode: (text: string) => number[]
    readonly decode: (tokenIds: readonly number[]) => string
    readonly vocabulary: readonly string[]
    readonly vocabularySize: number
}

export type LongCorpusStats = {
    readonly byteLength: number
    readonly characterCount: number
    readonly lineCount: number
}

export type LongCorpusText = {
    readonly filePath: string
    readonly rawText: string
    readonly stats: LongCorpusStats
}

export type LongCorpusPipelineOptions = {
    readonly contextLength?: number
    readonly batchSize?: number
    readonly validationRatio?: number
}

export type LongCorpusPipeline = {
    readonly rawText: string
    readonly stats: LongCorpusStats
    readonly vocabulary: readonly string[]
    readonly vocabularySize: number
    readonly tokenIds: readonly number[]
    readonly trainTokenIds: readonly number[]
    readonly validationTokenIds: readonly number[]
    readonly totalTokens: number
    readonly trainTokenCount: number
    readonly validationTokenCount: number
    readonly contextLength: number
    readonly batchSize: number
    readonly validationRatio: number
    readonly trainExampleCount: number
    readonly validationExampleCount: number
    readonly trainBatchCount: number
    readonly validationBatchCount: number
}

export type NextTokenBatchOptions = {
    readonly contextLength: number
    readonly batchSize: number
}

export type NextTokenBatch = {
    readonly batchIndex: number
    readonly startExampleIndex: number
    readonly inputTokenIds: readonly (readonly number[])[]
    readonly targetTokenIds: readonly number[]
}

export type TensorNextTokenBatch = {
    readonly inputTokenIds: tf.Tensor2D
    readonly targetTokenIds: tf.Tensor1D
}

export type PreparedLongCorpusDataset = {
    readonly version: 1
    readonly createdAt: string
    readonly sourceFilePath: string | undefined
    readonly stats: LongCorpusStats
    readonly vocabulary: readonly string[]
    readonly tokenIds: readonly number[]
    readonly trainTokenIds: readonly number[]
    readonly validationTokenIds: readonly number[]
    readonly options: {
        readonly contextLength: number
        readonly batchSize: number
        readonly validationRatio: number
    }
}

export type SavePreparedLongCorpusDatasetOptions = {
    readonly sourceFilePath?: string
}

const defaultContextLength = 128
const defaultBatchSize = 16
const defaultValidationRatio = 0.05
const preparedDatasetVersion = 1

export async function loadLongCorpusText(filePath: string): Promise<LongCorpusText> {
    const rawText = await readFile(filePath, 'utf8')

    return {
        filePath,
        rawText,
        stats: createLongCorpusStats(rawText),
    }
}

export function createLongCorpusPipeline(
    rawText: string,
    tokenizer: LongCorpusTokenizer,
    options: LongCorpusPipelineOptions = {},
): LongCorpusPipeline {
    const contextLength = options.contextLength ?? defaultContextLength
    const batchSize = options.batchSize ?? defaultBatchSize
    const validationRatio = options.validationRatio ?? defaultValidationRatio

    validatePositiveInteger(contextLength, 'contextLength')
    validatePositiveInteger(batchSize, 'batchSize')
    validateValidationRatio(validationRatio)

    const tokenIds = tokenizer.encode(rawText)
    const validationTokenCount = Math.floor(tokenIds.length * validationRatio)
    const trainTokenCount = tokenIds.length - validationTokenCount
    const trainTokenIds = tokenIds.slice(0, trainTokenCount)
    const validationTokenIds = tokenIds.slice(trainTokenCount)
    const trainExampleCount = estimateNextTokenExampleCount(trainTokenIds.length, contextLength)
    const validationExampleCount = estimateNextTokenExampleCount(
        validationTokenIds.length,
        contextLength,
    )

    return {
        batchSize,
        contextLength,
        rawText,
        stats: createLongCorpusStats(rawText),
        tokenIds,
        totalTokens: tokenIds.length,
        trainBatchCount: getBatchCount(trainExampleCount, batchSize),
        trainExampleCount,
        trainTokenCount,
        trainTokenIds,
        validationBatchCount: getBatchCount(validationExampleCount, batchSize),
        validationExampleCount,
        validationRatio,
        validationTokenCount,
        validationTokenIds,
        vocabulary: [...tokenizer.vocabulary],
        vocabularySize: tokenizer.vocabularySize,
    }
}

export function estimateNextTokenExampleCount(tokenCount: number, contextLength: number): number {
    validateNonNegativeInteger(tokenCount, 'tokenCount')
    validatePositiveInteger(contextLength, 'contextLength')

    return Math.max(0, tokenCount - contextLength)
}

export function getBatchCount(exampleCount: number, batchSize: number): number {
    validateNonNegativeInteger(exampleCount, 'exampleCount')
    validatePositiveInteger(batchSize, 'batchSize')

    return Math.ceil(exampleCount / batchSize)
}

export function* iterateNextTokenBatches(
    tokenIds: readonly number[],
    options: NextTokenBatchOptions,
): Generator<NextTokenBatch> {
    validatePositiveInteger(options.contextLength, 'contextLength')
    validatePositiveInteger(options.batchSize, 'batchSize')

    const exampleCount = estimateNextTokenExampleCount(tokenIds.length, options.contextLength)
    let batchIndex = 0

    for (
        let startExampleIndex = 0;
        startExampleIndex < exampleCount;
        startExampleIndex += options.batchSize
    ) {
        const inputTokenIds: number[][] = []
        const targetTokenIds: number[] = []
        const endExampleIndex = Math.min(startExampleIndex + options.batchSize, exampleCount)

        for (let exampleIndex = startExampleIndex; exampleIndex < endExampleIndex; exampleIndex++) {
            const targetIndex = exampleIndex + options.contextLength

            inputTokenIds.push(tokenIds.slice(exampleIndex, targetIndex))
            targetTokenIds.push(readTokenIdAt(tokenIds, targetIndex))
        }

        yield {
            batchIndex,
            inputTokenIds,
            startExampleIndex,
            targetTokenIds,
        }

        batchIndex++
    }
}

export function nextTokenBatchToTensors(batch: NextTokenBatch): TensorNextTokenBatch {
    if (batch.inputTokenIds.length === 0) {
        throw new Error('nextTokenBatchToTensors attend un batch non vide.')
    }

    const contextLength = readContextLength(batch)

    for (const [rowIndex, row] of batch.inputTokenIds.entries()) {
        if (row.length !== contextLength) {
            throw new Error(
                `Toutes les lignes inputTokenIds doivent avoir la même longueur. Ligne ${String(
                    rowIndex,
                )}: ${String(row.length)}, attendu: ${String(contextLength)}.`,
            )
        }
    }

    if (batch.targetTokenIds.length !== batch.inputTokenIds.length) {
        throw new Error(
            `targetTokenIds doit contenir ${String(
                batch.inputTokenIds.length,
            )} valeurs. Nombre reçu: ${String(batch.targetTokenIds.length)}.`,
        )
    }

    return {
        inputTokenIds: tf.tensor2d(
            batch.inputTokenIds.map((row) => [...row]),
            [batch.inputTokenIds.length, contextLength],
            'int32',
        ),
        targetTokenIds: tf.tensor1d([...batch.targetTokenIds], 'int32'),
    }
}

export function disposeTensorNextTokenBatch(batch: TensorNextTokenBatch): void {
    batch.inputTokenIds.dispose()
    batch.targetTokenIds.dispose()
}

export async function savePreparedLongCorpusDataset(
    pipeline: LongCorpusPipeline,
    filePath: string,
    options: SavePreparedLongCorpusDatasetOptions = {},
): Promise<void> {
    const preparedDataset: PreparedLongCorpusDataset = {
        createdAt: new Date().toISOString(),
        options: {
            batchSize: pipeline.batchSize,
            contextLength: pipeline.contextLength,
            validationRatio: pipeline.validationRatio,
        },
        sourceFilePath: options.sourceFilePath,
        stats: pipeline.stats,
        tokenIds: pipeline.tokenIds,
        trainTokenIds: pipeline.trainTokenIds,
        validationTokenIds: pipeline.validationTokenIds,
        version: preparedDatasetVersion,
        vocabulary: pipeline.vocabulary,
    }

    await mkdir(dirname(filePath), { recursive: true })
    await writeFile(filePath, `${JSON.stringify(preparedDataset, null, 2)}\n`, 'utf8')
}

export async function loadPreparedLongCorpusDataset(
    filePath: string,
): Promise<PreparedLongCorpusDataset> {
    const rawJson = await readFile(filePath, 'utf8')
    const parsedValue = JSON.parse(rawJson) as unknown

    return validatePreparedLongCorpusDataset(parsedValue)
}

function createLongCorpusStats(rawText: string): LongCorpusStats {
    return {
        byteLength: Buffer.byteLength(rawText, 'utf8'),
        characterCount: Array.from(rawText).length,
        lineCount: rawText.length === 0 ? 0 : rawText.split(/\r\n|\n|\r/u).length,
    }
}

function readContextLength(batch: NextTokenBatch): number {
    const firstRow = batch.inputTokenIds[0]

    if (firstRow === undefined) {
        throw new Error('Batch vide.')
    }

    return firstRow.length
}

function readTokenIdAt(tokenIds: readonly number[], index: number): number {
    const tokenId = tokenIds[index]

    if (tokenId === undefined) {
        throw new Error(`Token introuvable à l’index ${String(index)}.`)
    }

    return tokenId
}

function validatePreparedLongCorpusDataset(value: unknown): PreparedLongCorpusDataset {
    if (!isRecord(value)) {
        throw new Error('Le cache dataset doit contenir un objet JSON.')
    }

    if (value.version !== preparedDatasetVersion) {
        throw new Error(
            `Version de cache dataset invalide. Attendu: ${String(preparedDatasetVersion)}.`,
        )
    }

    assertString(value.createdAt, 'createdAt')
    assertOptionalString(value.sourceFilePath, 'sourceFilePath')
    assertStats(value.stats)
    assertStringArray(value.vocabulary, 'vocabulary')
    assertNumberArray(value.tokenIds, 'tokenIds')
    assertNumberArray(value.trainTokenIds, 'trainTokenIds')
    assertNumberArray(value.validationTokenIds, 'validationTokenIds')

    if (!isRecord(value.options)) {
        throw new Error('options doit être un objet.')
    }

    validatePositiveInteger(value.options.contextLength as number, 'options.contextLength')
    validatePositiveInteger(value.options.batchSize as number, 'options.batchSize')
    validateValidationRatio(value.options.validationRatio as number)

    return value as PreparedLongCorpusDataset
}

function assertStats(value: unknown): asserts value is LongCorpusStats {
    if (!isRecord(value)) {
        throw new Error('stats doit être un objet.')
    }

    validateNonNegativeInteger(value.byteLength as number, 'stats.byteLength')
    validateNonNegativeInteger(value.characterCount as number, 'stats.characterCount')
    validateNonNegativeInteger(value.lineCount as number, 'stats.lineCount')
}

function assertString(value: unknown, name: string): asserts value is string {
    if (typeof value !== 'string') {
        throw new Error(`${name} doit être une chaîne de caractères.`)
    }
}

function assertOptionalString(value: unknown, name: string): asserts value is string | undefined {
    if (value !== undefined && typeof value !== 'string') {
        throw new Error(`${name} doit être une chaîne de caractères ou undefined.`)
    }
}

function assertStringArray(value: unknown, name: string): asserts value is readonly string[] {
    if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
        throw new Error(`${name} doit être un tableau de chaînes de caractères.`)
    }
}

function assertNumberArray(value: unknown, name: string): asserts value is readonly number[] {
    if (!Array.isArray(value) || value.some((item) => !Number.isInteger(item) || item < 0)) {
        throw new Error(`${name} doit être un tableau d’entiers positifs ou nuls.`)
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null
}

function validateValidationRatio(validationRatio: number): void {
    if (!Number.isFinite(validationRatio) || validationRatio < 0 || validationRatio >= 1) {
        throw new Error(
            `validationRatio doit être un nombre fini supérieur ou égal à 0 et strictement inférieur à 1. Valeur reçue: ${String(
                validationRatio,
            )}.`,
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

function validateNonNegativeInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value < 0) {
        throw new Error(
            `${name} doit être un entier positif ou nul. Valeur reçue: ${String(value)}.`,
        )
    }
}
