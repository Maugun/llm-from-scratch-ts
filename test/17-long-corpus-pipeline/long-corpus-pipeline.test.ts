import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import {
    createLongCorpusPipeline,
    disposeTensorNextTokenBatch,
    estimateNextTokenExampleCount,
    getBatchCount,
    iterateNextTokenBatches,
    loadLongCorpusText,
    loadPreparedLongCorpusDataset,
    nextTokenBatchToTensors,
    savePreparedLongCorpusDataset,
} from '../../src/modules/17-long-corpus-pipeline/index.js'

let temporaryDirectory: string

beforeEach(async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), 'typescript-llm-long-corpus-'))
})

afterEach(async () => {
    await rm(temporaryDirectory, { force: true, recursive: true })
})

describe('loadLongCorpusText', () => {
    it('lit un fichier UTF-8 temporaire et calcule les stats', async () => {
        const filePath = join(temporaryDirectory, 'corpus.txt')

        await writeFile(filePath, 'bonjour\nmodèle\n', 'utf8')

        const corpus = await loadLongCorpusText(filePath)

        expect(corpus.filePath).toBe(filePath)
        expect(corpus.rawText).toBe('bonjour\nmodèle\n')
        expect(corpus.stats.byteLength).toBe(Buffer.byteLength(corpus.rawText, 'utf8'))
        expect(corpus.stats.characterCount).toBe(Array.from(corpus.rawText).length)
        expect(corpus.stats.lineCount).toBe(3)
    })
})

describe('estimateNextTokenExampleCount', () => {
    it('calcule correctement le nombre d’exemples next-token', () => {
        expect(estimateNextTokenExampleCount(10, 4)).toBe(6)
    })

    it('retourne 0 si la séquence est trop courte', () => {
        expect(estimateNextTokenExampleCount(3, 4)).toBe(0)
        expect(estimateNextTokenExampleCount(4, 4)).toBe(0)
    })
})

describe('createLongCorpusPipeline', () => {
    it('crée un split train/validation déterministe', () => {
        const rawText = 'abcdefghij'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 2,
            contextLength: 3,
            validationRatio: 0.3,
        })

        expect(pipeline.trainTokenIds).toEqual([0, 1, 2, 3, 4, 5, 6])
        expect(pipeline.validationTokenIds).toEqual([7, 8, 9])
        expect(pipeline.trainExampleCount).toBe(4)
        expect(pipeline.trainBatchCount).toBe(2)
    })

    it('rejette les options invalides', () => {
        const tokenizer = createCharacterTokenizer('abc')

        expect(() => createLongCorpusPipeline('abc', tokenizer, { contextLength: 0 })).toThrow(
            'contextLength doit être un entier strictement positif.',
        )
        expect(() => createLongCorpusPipeline('abc', tokenizer, { batchSize: 0 })).toThrow(
            'batchSize doit être un entier strictement positif.',
        )
        expect(() => createLongCorpusPipeline('abc', tokenizer, { validationRatio: 1 })).toThrow(
            'validationRatio doit être un nombre fini supérieur ou égal à 0',
        )
        expect(() => createLongCorpusPipeline('abc', tokenizer, { validationRatio: -0.1 })).toThrow(
            'validationRatio doit être un nombre fini supérieur ou égal à 0',
        )
    })

    it('ne matérialise pas tous les exemples dans le résultat de pipeline', () => {
        const rawText = 'abcdefghij'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 2,
            contextLength: 3,
        })

        expect('examples' in pipeline).toBe(false)
    })

    it('fonctionne avec tokenizer caractère et mini corpus versionné', async () => {
        const rawText = await readFile(join(process.cwd(), 'data', 'tiny-corpus.txt'), 'utf8')
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 4,
            contextLength: 8,
        })

        expect(pipeline.totalTokens).toBe(rawText.length)
        expect(pipeline.vocabularySize).toBe(tokenizer.vocabularySize)
        expect(pipeline.trainExampleCount).toBeGreaterThan(0)
    })
})

describe('iterateNextTokenBatches', () => {
    it('génère des batches avec les bons contextes et les bonnes cibles', () => {
        const batches = Array.from(
            iterateNextTokenBatches([0, 1, 2, 3, 4, 5], {
                batchSize: 2,
                contextLength: 2,
            }),
        )

        expect(batches).toEqual([
            {
                batchIndex: 0,
                inputTokenIds: [
                    [0, 1],
                    [1, 2],
                ],
                startExampleIndex: 0,
                targetTokenIds: [2, 3],
            },
            {
                batchIndex: 1,
                inputTokenIds: [
                    [2, 3],
                    [3, 4],
                ],
                startExampleIndex: 2,
                targetTokenIds: [4, 5],
            },
        ])
    })
})

describe('getBatchCount', () => {
    it('calcule correctement le nombre de batches', () => {
        expect(getBatchCount(0, 4)).toBe(0)
        expect(getBatchCount(1, 4)).toBe(1)
        expect(getBatchCount(8, 4)).toBe(2)
        expect(getBatchCount(9, 4)).toBe(3)
    })
})

describe('nextTokenBatchToTensors', () => {
    it('convertit un batch en tenseurs TensorFlow.js', () => {
        const tensorBatch = nextTokenBatchToTensors({
            batchIndex: 0,
            inputTokenIds: [
                [0, 1],
                [1, 2],
            ],
            startExampleIndex: 0,
            targetTokenIds: [2, 3],
        })

        expect(tensorBatch.inputTokenIds.shape).toEqual([2, 2])
        expect(tensorBatch.targetTokenIds.shape).toEqual([2])

        disposeTensorNextTokenBatch(tensorBatch)
    })
})

describe('cache dataset préparé', () => {
    it('sauvegarde et recharge un cache JSON pédagogique', async () => {
        const rawText = 'abcdefghij'
        const tokenizer = createCharacterTokenizer(rawText)
        const pipeline = createLongCorpusPipeline(rawText, tokenizer, {
            batchSize: 2,
            contextLength: 3,
            validationRatio: 0.2,
        })
        const filePath = join(temporaryDirectory, 'cache', 'dataset.json')

        await savePreparedLongCorpusDataset(pipeline, filePath, {
            sourceFilePath: 'data/private/long-corpus.txt',
        })

        const loadedDataset = await loadPreparedLongCorpusDataset(filePath)

        expect(loadedDataset.version).toBe(1)
        expect(loadedDataset.sourceFilePath).toBe('data/private/long-corpus.txt')
        expect(loadedDataset.tokenIds).toEqual(pipeline.tokenIds)
        expect(loadedDataset.trainTokenIds).toEqual(pipeline.trainTokenIds)
        expect(loadedDataset.validationTokenIds).toEqual(pipeline.validationTokenIds)
        expect(loadedDataset.options.contextLength).toBe(3)
    })
})
