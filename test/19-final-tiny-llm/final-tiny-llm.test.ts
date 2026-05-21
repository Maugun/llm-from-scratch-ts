import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { execFile } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'

import * as tf from '@tensorflow/tfjs'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createLongCorpusPipeline } from '../../src/modules/17-long-corpus-pipeline/index.js'
import {
    chatWithFinalTinyLlm,
    createFinalTinyLlm,
    decodeWithBpe,
    disposeFinalTinyLlm,
    encodeWithBpe,
    evaluateFinalTinyLlm,
    generateFinalTinyLlmText,
    loadBpeTokenizer,
    loadFinalTinyLlmCheckpoint,
    predictFinalTinyLlmNextToken,
    predictFinalTinyLlmProbabilities,
    saveBpeTokenizer,
    saveFinalTinyLlmCheckpoint,
    trainBpeTokenizer,
    trainFinalTinyLlm,
    type FinalTinyLlm,
} from '../../src/modules/19-final-tiny-llm/index.js'
import { resolveCheckpointVersionPlan } from '../../src/modules/18-small-real-model-training/index.js'

const execFileAsync = promisify(execFile)

let temporaryDirectory: string

beforeEach(async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), 'typescript-llm-final-tiny-llm-'))
})

afterEach(async () => {
    await rm(temporaryDirectory, { force: true, recursive: true })
})

describe('tokenizer BPE pédagogique', () => {
    it('s’entraîne de manière déterministe', () => {
        const left = trainBpeTokenizer('abababab', { vocabularySize: 4 })
        const right = trainBpeTokenizer('abababab', { vocabularySize: 4 })

        expect(left.vocabulary).toEqual(right.vocabulary)
        expect(left.merges).toEqual(right.merges)
    })

    it('notifie la progression pendant les merges BPE', () => {
        const progressRatios: number[] = []

        trainBpeTokenizer('abababab', {
            onProgress: (progress) => {
                progressRatios.push(progress.progressRatio)
            },
            vocabularySize: 4,
        })

        expect(progressRatios.length).toBeGreaterThan(0)
        expect(progressRatios.at(-1)).toBe(1)
    })

    it('encode puis décode sans perte sur un texte connu', () => {
        const tokenizer = trainBpeTokenizer('bonjour bonjour', { vocabularySize: 10 })
        const tokenIds = encodeWithBpe(tokenizer, 'bonjour')

        expect(decodeWithBpe(tokenizer, tokenIds)).toBe('bonjour')
    })

    it('sauvegarde puis recharge le tokenizer', async () => {
        const tokenizer = trainBpeTokenizer('abababab', { vocabularySize: 4 })
        const filePath = join(temporaryDirectory, 'tokenizer.json')

        await saveBpeTokenizer(tokenizer, filePath)
        const loaded = await loadBpeTokenizer(filePath)

        expect(loaded.vocabulary).toEqual(tokenizer.vocabulary)
        expect(loaded.merges).toEqual(tokenizer.merges)
        expect(loaded.decode(loaded.encode('abab'))).toBe('abab')
    })

    it('rejette un caractère absent du vocabulaire', () => {
        const tokenizer = trainBpeTokenizer('abcabc', { vocabularySize: 3 })

        expect(() => tokenizer.encode('abcd')).toThrow('Caractère absent du vocabulaire BPE')
    })
})

describe('modèle final', () => {
    it('crée un modèle avec les bonnes shapes', () => {
        const model = createTinyModel()

        try {
            expect(model.tokenEmbeddings.shape).toEqual([5, 8])
            expect(model.positionEmbeddings.shape).toEqual([3, 8])
            expect(model.layers).toHaveLength(1)
            expect(model.layers[0]?.queryWeights.shape).toEqual([8, 8])
            expect(model.layers[0]?.layerNorm1Scale.shape).toEqual([8])
            expect(model.outputWeights.shape).toEqual([8, 5])
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('valide les options du modèle', () => {
        expect(() =>
            createFinalTinyLlm({
                contextLength: 3,
                embeddingDimension: 7,
                feedForwardDimension: 16,
                headCount: 4,
                layerCount: 1,
                vocabularySize: 5,
            }),
        ).toThrow('embeddingDimension doit être divisible par headCount')
    })

    it('retourne une distribution normalisée', () => {
        const model = createTinyModel()

        try {
            const probabilities = predictFinalTinyLlmProbabilities(model, [0, 1, 2])
            const sum = probabilities.reduce((total, probability) => total + probability, 0)

            expect(probabilities).toHaveLength(5)
            expect(sum).toBeCloseTo(1, 5)
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('rejette un contexte de mauvaise taille ou hors vocabulaire', () => {
        const model = createTinyModel()

        try {
            expect(() => predictFinalTinyLlmProbabilities(model, [0, 1])).toThrow(
                'Le contexte doit contenir 3 tokens',
            )
            expect(() => predictFinalTinyLlmProbabilities(model, [0, 1, 99])).toThrow(
                'inputTokenIds[2] doit être un entier entre 0 et 4',
            )
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('retourne un token prédit valide', () => {
        const model = createTinyModel()

        try {
            const tokenId = predictFinalTinyLlmNextToken(model, [0, 1, 2])

            expect(tokenId).toBeGreaterThanOrEqual(0)
            expect(tokenId).toBeLessThan(model.vocabularySize)
        } finally {
            disposeFinalTinyLlm(model)
        }
    })
})

describe('évaluation, entraînement et génération', () => {
    it('calcule une loss finie et réduit la loss sur un mini corpus répétitif', () => {
        const rawText = 'abababababababababababababababababababab'
        const tokenizer = trainBpeTokenizer(rawText, { vocabularySize: 2 })
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 4,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const model = createTinyModel(2)

        try {
            const before = evaluateFinalTinyLlm(model, pipeline.validationTokenIds, {
                batchSize: 4,
                maxBatches: 2,
            })
            const history = trainFinalTinyLlm(model, pipeline, {
                batchSize: 4,
                epochs: 8,
                learningRate: 0.02,
                maxTrainBatchesPerEpoch: 6,
                maxValidationBatches: 2,
            })

            expect(Number.isFinite(before.averageLoss)).toBe(true)
            expect(history.finalValidationLoss).toBeLessThan(history.initialValidationLoss)
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('restaure les meilleurs poids du run quand saveBestEpochOnly est actif', () => {
        const rawText = 'abababababababababababababababababababab'
        const tokenizer = trainBpeTokenizer(rawText, { vocabularySize: 2 })
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 4,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const model = createTinyModel(2)

        try {
            const history = trainFinalTinyLlm(model, pipeline, {
                batchSize: 4,
                epochs: 3,
                learningRate: 0.02,
                maxTrainBatchesPerEpoch: 4,
                maxValidationBatches: 2,
                saveBestEpochOnly: true,
            })

            expect(history.restoredBestEpochWeights).toBe(true)
            expect(history.finalValidationLoss).toBeCloseTo(history.bestValidationLoss, 5)
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('génère un texte qui commence par le prompt', () => {
        const tokenizer = trainBpeTokenizer('abcdeabcde', { vocabularySize: 5 })
        const model = createTinyModel(5)

        try {
            forceSimpleOutputDistribution(model, [0, 3, 1, -1, -2])

            const result = generateFinalTinyLlmText(model, tokenizer, 'abc', {
                maxNewTokens: 4,
                strategy: 'greedy',
            })

            expect(result.text.startsWith('abc')).toBe(true)
            expect(result.generatedTokenIds).toHaveLength(4)
            expect(result.steps).toHaveLength(4)
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('formate un chat playground comme une génération de texte', () => {
        const tokenizer = trainBpeTokenizer('Utilisateur: Bonjour\nAssistant: Salut', {
            vocabularySize: 20,
        })
        const model = createFinalTinyLlm({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            headCount: 4,
            layerCount: 1,
            seed: 123,
            vocabularySize: tokenizer.vocabularySize,
        })

        try {
            const result = chatWithFinalTinyLlm(
                model,
                tokenizer,
                [{ content: 'Bonjour', role: 'user' }],
                { maxNewTokens: 2, strategy: 'greedy' },
            )

            expect(result.prompt).toContain('Utilisateur: Bonjour')
            expect(result.prompt).toContain('Assistant:')
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('n’impose pas les labels de chat si le tokenizer ne les connaît pas', () => {
        const tokenizer = trainBpeTokenizer('bonjour bonjour', { vocabularySize: 10 })
        const model = createFinalTinyLlm({
            contextLength: 1,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            headCount: 4,
            layerCount: 1,
            seed: 123,
            vocabularySize: tokenizer.vocabularySize,
        })

        try {
            const result = chatWithFinalTinyLlm(
                model,
                tokenizer,
                [{ content: 'bonjour', role: 'user' }],
                { maxNewTokens: 2, strategy: 'greedy' },
            )

            expect(result.prompt).toBe('bonjour')
        } finally {
            disposeFinalTinyLlm(model)
        }
    })
})

describe('checkpoint final', () => {
    it('sauvegarde puis recharge modèle et tokenizer', async () => {
        const tokenizer = trainBpeTokenizer('abcdeabcde', { vocabularySize: 5 })
        const model = createTinyModel(5)
        const checkpointPath = join(temporaryDirectory, 'checkpoint')

        try {
            forceSimpleOutputDistribution(model, [0, 3, 1, -1, -2])
            await saveFinalTinyLlmCheckpoint(model, tokenizer, checkpointPath, {
                extra: { test: true },
            })

            const loaded = await loadFinalTinyLlmCheckpoint(checkpointPath)

            try {
                expect(loaded.tokenizer.vocabulary).toEqual(tokenizer.vocabulary)
                expect(predictFinalTinyLlmProbabilities(loaded.model, [0, 1, 2])).toEqual(
                    predictFinalTinyLlmProbabilities(model, [0, 1, 2]),
                )
            } finally {
                disposeFinalTinyLlm(loaded.model)
            }
        } finally {
            disposeFinalTinyLlm(model)
        }
    })

    it('utilise le versioning partagé des checkpoints', async () => {
        const checkpointPath = join(temporaryDirectory, 'versions')

        await mkdir(join(checkpointPath, 'v1'), { recursive: true })
        await mkdir(join(checkpointPath, 'v2'), { recursive: true })
        await saveMetadataPlaceholder(join(checkpointPath, 'v1'))
        await saveMetadataPlaceholder(join(checkpointPath, 'v2'))

        const plan = await resolveCheckpointVersionPlan({
            checkpointPath,
            continueTrain: true,
        })

        expect(plan.loadVersion?.versionName).toBe('v2')
        expect(plan.saveVersion.versionName).toBe('v3')
    })
})

describe('CLI/config module 19', () => {
    it('accepte une config qui omet les booléens optionnels', async () => {
        const configPath = join(temporaryDirectory, 'config.json')

        await saveJson(configPath, {
            batchOrder: 'shuffled',
            batchSize: 4,
            bpeMaxTrainingCharacters: 1000,
            bpeVocabularySize: 32,
            checkpointPath: join(temporaryDirectory, 'checkpoints'),
            contextLength: 3,
            corpusPath: join(process.cwd(), 'data', 'tiny-corpus.txt'),
            embeddingDimension: 8,
            epochs: 1,
            feedForwardDimension: 16,
            headCount: 4,
            layerCount: 1,
            learningRate: 0.001,
            maxNewTokens: 2,
            maxTrainBatchesPerEpoch: 1,
            maxValidationBatches: 1,
            prompt: 'bonjour',
            seed: 19,
            shuffleSeed: 19,
            strategy: 'greedy',
            temperature: 1,
            topK: 2,
            validationRatio: 0.4,
        })

        const result = await execFileAsync('node', [
            '--import',
            'tsx',
            'src/modules/19-final-tiny-llm/demo.ts',
            '--mode',
            'generate',
            '--config',
            configPath,
        ])

        expect(result.stdout).toContain('saveBestEpochOnly: false')
        expect(result.stdout).toContain('skipCheckpointWhenNoImprovement: false')
    })
})

function createTinyModel(vocabularySize = 5): FinalTinyLlm {
    return createFinalTinyLlm({
        contextLength: 3,
        embeddingDimension: 8,
        feedForwardDimension: 16,
        headCount: 4,
        layerCount: 1,
        seed: 123,
        vocabularySize,
    })
}

function forceSimpleOutputDistribution(
    model: FinalTinyLlm,
    outputBiasValues: readonly number[],
): void {
    const outputWeights = tf.zeros([model.embeddingDimension, model.vocabularySize])
    const outputBias = tf.tensor1d([...outputBiasValues])

    model.outputWeights.assign(outputWeights)
    model.outputBias.assign(outputBias)
    outputWeights.dispose()
    outputBias.dispose()
}

async function saveMetadataPlaceholder(directoryPath: string): Promise<void> {
    const { writeFile } = await import('node:fs/promises')

    await writeFile(join(directoryPath, 'metadata.json'), '{}')
}

async function saveJson(filePath: string, value: unknown): Promise<void> {
    const { writeFile } = await import('node:fs/promises')

    await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8')
}
