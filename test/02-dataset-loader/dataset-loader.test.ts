import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
    createTokenDataset,
    loadTextFile,
    loadTokenDatasetFromFile,
} from '../../src/modules/02-dataset-loader/index.js'
import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'

let temporaryDirectory: string

beforeEach(async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), 'typescript-llm-dataset-'))
})

afterEach(async () => {
    await rm(temporaryDirectory, { force: true, recursive: true })
})

describe('loadTextFile', () => {
    it('lit correctement un fichier .txt UTF-8 temporaire', async () => {
        const filePath = join(temporaryDirectory, 'sample.txt')

        await writeFile(filePath, 'bonjour modèle\n', 'utf8')

        await expect(loadTextFile(filePath)).resolves.toBe('bonjour modèle\n')
    })

    it("échoue avec l'erreur Node standard si le fichier n'existe pas", async () => {
        const filePath = join(temporaryDirectory, 'missing.txt')

        await expect(loadTextFile(filePath)).rejects.toMatchObject({ code: 'ENOENT' })
    })
})

describe('createTokenDataset', () => {
    it('encode le texte avec le tokenizer fourni', () => {
        const rawText = 'abcde'
        const tokenizer = createCharacterTokenizer(rawText)

        const dataset = createTokenDataset(rawText, tokenizer, { validationRatio: 0.4 })

        expect(dataset.tokenIds).toEqual([0, 1, 2, 3, 4])
    })

    it('crée un split train/validation déterministe', () => {
        const rawText = 'abcdefghij'
        const tokenizer = createCharacterTokenizer(rawText)

        const dataset = createTokenDataset(rawText, tokenizer, { validationRatio: 0.3 })

        expect(dataset.trainTokenIds).toEqual([0, 1, 2, 3, 4, 5, 6])
        expect(dataset.validationTokenIds).toEqual([7, 8, 9])
    })

    it('expose les compteurs de tokens', () => {
        const rawText = 'abcdefghij'
        const tokenizer = createCharacterTokenizer(rawText)

        const dataset = createTokenDataset(rawText, tokenizer, { validationRatio: 0.2 })

        expect(dataset.totalTokens).toBe(10)
        expect(dataset.trainTokenCount).toBe(8)
        expect(dataset.validationTokenCount).toBe(2)
    })

    it('accepte validationRatio: 0 pour tout garder en train', () => {
        const rawText = 'abc'
        const tokenizer = createCharacterTokenizer(rawText)

        const dataset = createTokenDataset(rawText, tokenizer, { validationRatio: 0 })

        expect(dataset.trainTokenIds).toEqual([0, 1, 2])
        expect(dataset.validationTokenIds).toEqual([])
    })

    it('rejette un validationRatio invalide', () => {
        const tokenizer = createCharacterTokenizer('abc')

        expect(() => createTokenDataset('abc', tokenizer, { validationRatio: 1 })).toThrow(
            'validationRatio doit être un nombre fini supérieur ou égal à 0 et strictement inférieur à 1.',
        )
        expect(() => createTokenDataset('abc', tokenizer, { validationRatio: -0.1 })).toThrow(
            'validationRatio doit être un nombre fini supérieur ou égal à 0 et strictement inférieur à 1.',
        )
        expect(() => createTokenDataset('abc', tokenizer, { validationRatio: Number.NaN })).toThrow(
            'validationRatio doit être un nombre fini supérieur ou égal à 0 et strictement inférieur à 1.',
        )
    })

    it('propage clairement les erreurs du tokenizer', () => {
        const tokenizer = createCharacterTokenizer('abc')

        expect(() => createTokenDataset('abcd', tokenizer)).toThrow(
            'Impossible d\'encoder le caractère inconnu "d".',
        )
    })
})

describe('loadTokenDatasetFromFile', () => {
    it('combine lecture fichier et encodage', async () => {
        const filePath = join(temporaryDirectory, 'sample.txt')
        const rawText = 'bonjour\n'
        const tokenizer = createCharacterTokenizer(rawText)

        await writeFile(filePath, rawText, 'utf8')

        const dataset = await loadTokenDatasetFromFile(filePath, tokenizer, {
            validationRatio: 0.25,
        })

        expect(dataset.rawText).toBe(rawText)
        expect(dataset.totalTokens).toBe(rawText.length)
        expect(dataset.trainTokenCount).toBe(6)
        expect(dataset.validationTokenCount).toBe(2)
    })

    it('charge et encode le mini corpus versionné', async () => {
        const filePath = join(process.cwd(), 'data', 'tiny-corpus.txt')
        const rawText = await loadTextFile(filePath)
        const tokenizer = createCharacterTokenizer(rawText)

        const dataset = await loadTokenDatasetFromFile(filePath, tokenizer, {
            validationRatio: 0.1,
        })

        expect(dataset.rawText).toBe(rawText)
        expect(dataset.totalTokens).toBe(rawText.length)
        expect(dataset.trainTokenCount + dataset.validationTokenCount).toBe(dataset.totalTokens)
        expect(dataset.validationTokenCount).toBe(Math.floor(dataset.totalTokens * 0.1))
    })
})
