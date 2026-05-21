import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import * as tf from '@tensorflow/tfjs'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createLongCorpusPipeline } from '../../src/modules/17-long-corpus-pipeline/index.js'
import {
    createSmallLanguageModel,
    disposeSmallLanguageModel,
    evaluateSmallLanguageModel,
    generateSmallLanguageModelText,
    listCheckpointVersions,
    loadSmallLanguageModelCheckpoint,
    predictSmallLanguageModelProbabilities,
    resolveCheckpointVersionPlan,
    saveSmallLanguageModelCheckpoint,
    selectSmallLanguageModelNextToken,
    trainSmallLanguageModel,
    type SmallLanguageModel,
} from '../../src/modules/18-small-real-model-training/index.js'

let temporaryDirectory: string

beforeEach(async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), 'typescript-llm-small-real-model-'))
})

afterEach(async () => {
    await rm(temporaryDirectory, { force: true, recursive: true })
})

describe('versioning des checkpoints', () => {
    it('liste seulement les dossiers de version qui contiennent un metadata.json', async () => {
        const checkpointPath = join(temporaryDirectory, 'versions')

        await createCheckpointVersionFolder(checkpointPath, 'v2')
        await createCheckpointVersionFolder(checkpointPath, 'v1')
        await mkdir(join(checkpointPath, 'draft'), { recursive: true })
        await mkdir(join(checkpointPath, 'v3'), { recursive: true })

        const versions = await listCheckpointVersions(checkpointPath)

        expect(versions.map((version) => version.versionName)).toEqual(['v1', 'v2'])
    })

    it('charge la dernière version disponible si aucune version précise n’est demandée', async () => {
        const checkpointPath = join(temporaryDirectory, 'versions')

        await createCheckpointVersionFolder(checkpointPath, 'v1')
        await createCheckpointVersionFolder(checkpointPath, 'v3')

        const plan = await resolveCheckpointVersionPlan({ checkpointPath })

        expect(plan.loadVersion?.versionName).toBe('v3')
        expect(plan.saveVersion.versionName).toBe('v3')
    })

    it('continue depuis la version demandée puis sauvegarde dans la prochaine version libre', async () => {
        const checkpointPath = join(temporaryDirectory, 'versions')

        await createCheckpointVersionFolder(checkpointPath, 'v1')
        await createCheckpointVersionFolder(checkpointPath, 'v2')
        await createCheckpointVersionFolder(checkpointPath, 'v4')

        const plan = await resolveCheckpointVersionPlan({
            checkpointPath,
            checkpointVersion: 'v2',
            continueTrain: true,
        })

        expect(plan.loadVersion?.versionName).toBe('v2')
        expect(plan.saveVersion.versionName).toBe('v5')
    })

    it('force un nouvel entraînement dans la prochaine version libre', async () => {
        const checkpointPath = join(temporaryDirectory, 'versions')

        await createCheckpointVersionFolder(checkpointPath, 'v1')
        await createCheckpointVersionFolder(checkpointPath, 'v2')

        const plan = await resolveCheckpointVersionPlan({
            checkpointPath,
            checkpointVersion: 'v1',
            forceTrain: true,
        })

        expect(plan.loadVersion).toBeUndefined()
        expect(plan.saveVersion.versionName).toBe('v3')
    })

    it('crée v1 quand aucun checkpoint versionné n’existe', async () => {
        const checkpointPath = join(temporaryDirectory, 'versions')
        const plan = await resolveCheckpointVersionPlan({ checkpointPath })

        expect(plan.loadVersion).toBeUndefined()
        expect(plan.saveVersion.versionName).toBe('v1')
    })

    it('rejette une version de checkpoint invalide', async () => {
        await expect(
            resolveCheckpointVersionPlan({
                checkpointPath: temporaryDirectory,
                checkpointVersion: '../v1',
            }),
        ).rejects.toThrow('checkpointVersion doit être au format')
    })
})

describe('createSmallLanguageModel', () => {
    it('crée un modèle avec les bonnes shapes', () => {
        const model = createTinyModel()

        try {
            expect(model.tokenEmbeddings.shape).toEqual([5, 4])
            expect(model.positionEmbeddings.shape).toEqual([3, 4])
            expect(model.layers).toHaveLength(1)
            expect(model.layers[0]?.feedForwardInputWeights.shape).toEqual([4, 8])
            expect(model.outputWeights.shape).toEqual([4, 5])
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('rejette les options invalides', () => {
        expect(() =>
            createSmallLanguageModel({
                contextLength: 3,
                embeddingDimension: 4,
                feedForwardDimension: 8,
                layerCount: 1,
                vocabularySize: 0,
            }),
        ).toThrow('vocabularySize doit être un entier strictement positif')

        expect(() =>
            createSmallLanguageModel({
                contextLength: 0,
                embeddingDimension: 4,
                feedForwardDimension: 8,
                layerCount: 1,
                vocabularySize: 5,
            }),
        ).toThrow('contextLength doit être un entier strictement positif')
    })
})

describe('prédiction et sampling', () => {
    it('retourne une distribution normalisée', () => {
        const model = createTinyModel()

        try {
            const probabilities = predictSmallLanguageModelProbabilities(model, [0, 1, 2])
            const sum = probabilities.reduce((total, probability) => total + probability, 0)

            expect(probabilities).toHaveLength(5)
            expect(sum).toBeCloseTo(1, 5)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('rejette un contexte de mauvaise taille', () => {
        const model = createTinyModel()

        try {
            expect(() => predictSmallLanguageModelProbabilities(model, [0, 1])).toThrow(
                'Le contexte doit contenir 3 tokens',
            )
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('rejette un token hors vocabulaire', () => {
        const model = createTinyModel()

        try {
            expect(() => predictSmallLanguageModelProbabilities(model, [0, 1, 99])).toThrow(
                'inputTokenIds[2] doit être un entier entre 0 et 4',
            )
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('utilise greedy, température et top-k via les stratégies du module 11', () => {
        const model = createTinyModel()

        try {
            forceSimpleOutputDistribution(model, [0, 3, 1, -1, -2])

            expect(
                selectSmallLanguageModelNextToken(model, [0, 1, 2], { strategy: 'greedy' }),
            ).toBe(1)
            expect(
                selectSmallLanguageModelNextToken(model, [0, 1, 2], {
                    random: () => 0,
                    strategy: 'temperature',
                    temperature: 1,
                }),
            ).toBe(0)
            expect(
                selectSmallLanguageModelNextToken(model, [0, 1, 2], {
                    random: () => 0.99,
                    strategy: 'topK',
                    topK: 2,
                }),
            ).toBe(2)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })
})

describe('génération', () => {
    it('retourne un texte qui commence par le prompt et respecte maxNewTokens', () => {
        const model = createTinyModel()
        const tokenizer = createCharacterTokenizer('abcde')

        try {
            forceSimpleOutputDistribution(model, [0, 3, 1, -1, -2])

            const result = generateSmallLanguageModelText(model, tokenizer, 'abc', {
                maxNewTokens: 4,
                strategy: 'greedy',
            })

            expect(result.text.startsWith('abc')).toBe(true)
            expect(result.generatedTokenIds).toHaveLength(4)
            expect(result.steps).toHaveLength(4)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })
})

describe('évaluation et entraînement', () => {
    it('calcule une loss et une perplexité finies', () => {
        const rawText = 'abababababababababab'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 2,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const model = createSmallLanguageModel({
            contextLength: 3,
            embeddingDimension: 4,
            feedForwardDimension: 8,
            layerCount: 1,
            vocabularySize: tokenizer.vocabularySize,
        })

        try {
            const evaluation = evaluateSmallLanguageModel(model, pipeline.validationTokenIds, {
                batchSize: 2,
            })

            expect(Number.isFinite(evaluation.averageLoss)).toBe(true)
            expect(Number.isFinite(evaluation.perplexity)).toBe(true)
            expect(evaluation.evaluatedBatches).toBeGreaterThan(0)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('réduit la loss sur un corpus répétitif', () => {
        const rawText = 'abababababababababababababababababababab'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 4,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const model = createSmallLanguageModel({
            contextLength: 3,
            embeddingDimension: 4,
            feedForwardDimension: 8,
            layerCount: 1,
            vocabularySize: tokenizer.vocabularySize,
        })

        try {
            const history = trainSmallLanguageModel(model, pipeline, {
                batchSize: 4,
                epochs: 8,
                learningRate: 0.05,
                maxTrainBatchesPerEpoch: 6,
                maxValidationBatches: 2,
            })

            expect(history.finalValidationLoss).toBeLessThan(history.initialValidationLoss)
            expect(history.epochs).toHaveLength(8)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('notifie la progression pendant l’entraînement', () => {
        const rawText = 'abababababababababababababababababababab'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 4,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const model = createSmallLanguageModel({
            contextLength: 3,
            embeddingDimension: 4,
            feedForwardDimension: 8,
            layerCount: 1,
            vocabularySize: tokenizer.vocabularySize,
        })
        const progressRatios: number[] = []

        try {
            trainSmallLanguageModel(model, pipeline, {
                batchSize: 4,
                epochs: 1,
                learningRate: 0.05,
                maxTrainBatchesPerEpoch: 3,
                maxValidationBatches: 1,
                onProgress: (progress) => {
                    progressRatios.push(progress.progressRatio)
                },
            })

            expect(progressRatios).toHaveLength(3)
            expect(progressRatios.at(-1)).toBe(1)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('rejette un batchOrder invalide', () => {
        const rawText = 'abababababababababab'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 2,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const model = createSmallLanguageModel({
            contextLength: 3,
            embeddingDimension: 4,
            feedForwardDimension: 8,
            layerCount: 1,
            vocabularySize: tokenizer.vocabularySize,
        })

        try {
            expect(() =>
                trainSmallLanguageModel(model, pipeline, {
                    batchOrder: 'random' as 'shuffled',
                    epochs: 1,
                    learningRate: 0.05,
                    maxTrainBatchesPerEpoch: 1,
                    maxValidationBatches: 1,
                }),
            ).toThrow('batchOrder doit valoir "sequential" ou "shuffled"')
        } finally {
            disposeSmallLanguageModel(model)
        }
    })

    it('fonctionne avec le mini corpus versionné', async () => {
        const rawText = await readFile(join(process.cwd(), 'data', 'tiny-corpus.txt'), 'utf8')
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 2,
            contextLength: 4,
            validationRatio: 0.2,
        })
        const model = createSmallLanguageModel({
            contextLength: 4,
            embeddingDimension: 4,
            feedForwardDimension: 8,
            layerCount: 1,
            vocabularySize: tokenizer.vocabularySize,
        })

        try {
            const history = trainSmallLanguageModel(model, pipeline, {
                batchSize: 2,
                epochs: 1,
                learningRate: 0.01,
                maxTrainBatchesPerEpoch: 2,
                maxValidationBatches: 1,
            })

            expect(history.epochs).toHaveLength(1)
        } finally {
            disposeSmallLanguageModel(model)
        }
    })
})

describe('checkpoint', () => {
    it('sauvegarde puis recharge un modèle avec des options cohérentes', async () => {
        const model = createTinyModel()
        const checkpointPath = join(temporaryDirectory, 'checkpoint')

        try {
            forceSimpleOutputDistribution(model, [0, 3, 1, -1, -2])

            await saveSmallLanguageModelCheckpoint(model, checkpointPath, {
                extra: { test: true },
            })

            const loadedModel = await loadSmallLanguageModelCheckpoint(checkpointPath)

            try {
                expect(loadedModel.contextLength).toBe(model.contextLength)
                expect(loadedModel.vocabularySize).toBe(model.vocabularySize)
                expect(predictSmallLanguageModelProbabilities(loadedModel, [0, 1, 2])).toEqual(
                    predictSmallLanguageModelProbabilities(model, [0, 1, 2]),
                )
            } finally {
                disposeSmallLanguageModel(loadedModel)
            }
        } finally {
            disposeSmallLanguageModel(model)
        }
    })
})

function createTinyModel(): SmallLanguageModel {
    return createSmallLanguageModel({
        contextLength: 3,
        embeddingDimension: 4,
        feedForwardDimension: 8,
        layerCount: 1,
        seed: 123,
        vocabularySize: 5,
    })
}

function forceSimpleOutputDistribution(
    model: SmallLanguageModel,
    outputBiasValues: readonly number[],
): void {
    const outputWeights = tf.zeros([model.embeddingDimension, model.vocabularySize])
    const outputBias = tf.tensor1d([...outputBiasValues])

    model.outputWeights.assign(outputWeights)
    model.outputBias.assign(outputBias)
    outputWeights.dispose()
    outputBias.dispose()
}

async function createCheckpointVersionFolder(
    checkpointPath: string,
    versionName: string,
): Promise<void> {
    const versionDirectoryPath = join(checkpointPath, versionName)

    await mkdir(versionDirectoryPath, { recursive: true })
    await writeFile(join(versionDirectoryPath, 'metadata.json'), '{}')
}
