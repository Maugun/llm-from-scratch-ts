import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import { createNextTokenExamples } from '../../src/modules/08-training-loop-cpu/index.js'
import {
    createMinimalLanguageModel,
    trainMinimalLanguageModel,
} from '../../src/modules/09-minimal-trainable-language-model/index.js'
import {
    generateText,
    generateTokenIds,
    getGenerationContext,
} from '../../src/modules/10-text-generation/index.js'

describe('getGenerationContext', () => {
    it('retourne les derniers contextLength tokens', () => {
        expect(getGenerationContext([1, 2, 3, 4], 2)).toEqual([3, 4])
    })

    it('rejette une séquence trop courte', () => {
        expect(() => getGenerationContext([1], 2)).toThrow(
            'La séquence doit contenir au moins 2 tokens pour construire le contexte. Nombre reçu: 1.',
        )
    })
})

describe('generateTokenIds', () => {
    it('ajoute exactement maxNewTokens tokens', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })
        model.outputBias[1] = 3

        const result = generateTokenIds(model, [0, 2], { maxNewTokens: 4 })

        expect(result.generatedTokenIds).toHaveLength(4)
        expect(result.tokenIds).toHaveLength(6)
    })

    it('retourne une étape par token généré', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })
        model.outputBias[1] = 3

        const result = generateTokenIds(model, [0, 2], { maxNewTokens: 3 })

        expect(result.steps).toHaveLength(3)
    })

    it('inclut le contexte utilisé et le token prédit dans chaque étape', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })
        model.outputBias[1] = 3

        const result = generateTokenIds(model, [0, 2], { maxNewTokens: 1 })

        expect(result.steps[0]).toMatchObject({
            contextTokenIds: [0, 2],
            predictedTokenId: 1,
            step: 1,
        })
    })

    it('rejette maxNewTokens invalide', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })

        expect(() => generateTokenIds(model, [0, 1], { maxNewTokens: 0 })).toThrow(
            'maxNewTokens doit être un entier strictement positif.',
        )
    })
})

describe('generateText', () => {
    it('retourne un texte qui commence par le prompt', () => {
        const tokenizer = createCharacterTokenizer('abc')
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: tokenizer.vocabularySize,
        })

        const result = generateText(model, tokenizer, 'ab', { maxNewTokens: 2 })

        expect(result.text.startsWith('ab')).toBe(true)
    })

    it('propage les erreurs du tokenizer pour un caractère inconnu', () => {
        const tokenizer = createCharacterTokenizer('abc')
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: tokenizer.vocabularySize,
        })

        expect(() => generateText(model, tokenizer, 'az', { maxNewTokens: 1 })).toThrow(
            'Impossible d\'encoder le caractère inconnu "z".',
        )
    })

    it('fonctionne avec tokenizer, mini corpus, dataset loader et modèle entraîné', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength: 4 })
        const model = createMinimalLanguageModel({
            contextLength: 4,
            vocabularySize: tokenizer.vocabularySize,
        })

        trainMinimalLanguageModel(model, examples, {
            epochs: 2,
            learningRate: 0.2,
        })

        const result = generateText(model, tokenizer, 'bonj', { maxNewTokens: 3 })

        expect(result.text.startsWith('bonj')).toBe(true)
        expect(result.generatedTokenIds).toHaveLength(3)
        expect(result.steps).toHaveLength(3)
    })
})
