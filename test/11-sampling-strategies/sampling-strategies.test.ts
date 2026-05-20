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
    applyTemperature,
    filterTopK,
    generateTextWithSampling,
    generateTokenIdsWithSampling,
    sampleFromProbabilities,
    selectNextToken,
} from '../../src/modules/11-sampling-strategies/index.js'

describe('applyTemperature', () => {
    it('conserve une distribution normalisée', () => {
        const probabilities = applyTemperature([0.2, 0.3, 0.5], 0.7)
        const sum = probabilities.reduce((total, probability) => total + probability, 0)

        expect(sum).toBeCloseTo(1)
    })

    it('rend la distribution plus concentrée avec une température basse', () => {
        const originalProbabilities = [0.2, 0.3, 0.5]
        const adjustedProbabilities = applyTemperature(originalProbabilities, 0.5)

        expect(adjustedProbabilities[2]).toBeGreaterThan(originalProbabilities[2] ?? 0)
    })

    it('rejette une température invalide', () => {
        expect(() => applyTemperature([0.5, 0.5], 0)).toThrow(
            'temperature doit être un nombre fini strictement positif.',
        )
    })
})

describe('filterTopK', () => {
    it('garde seulement les k plus grandes probabilités', () => {
        const probabilities = filterTopK([0.1, 0.7, 0.2], 2)

        expect(probabilities[0]).toBe(0)
        expect(probabilities[1]).toBeCloseTo(0.7 / 0.9)
        expect(probabilities[2]).toBeCloseTo(0.2 / 0.9)
    })

    it('renormalise la distribution', () => {
        const probabilities = filterTopK([0.1, 0.7, 0.2], 2)
        const sum = probabilities.reduce((total, probability) => total + probability, 0)

        expect(sum).toBeCloseTo(1)
    })

    it('rejette topK invalide', () => {
        expect(() => filterTopK([0.5, 0.5], 0)).toThrow(
            'topK doit être un entier strictement positif.',
        )
    })
})

describe('sampleFromProbabilities', () => {
    it('est déterministe avec une fonction random contrôlée', () => {
        expect(sampleFromProbabilities([0.25, 0.75], () => 0.5)).toBe(1)
        expect(sampleFromProbabilities([0.25, 0.75], () => 0.1)).toBe(0)
    })
})

describe('selectNextToken', () => {
    it('retourne le meilleur token en mode greedy', () => {
        expect(selectNextToken([0.1, 0.8, 0.1], { strategy: 'greedy' })).toBe(1)
    })
})

describe('generateTokenIdsWithSampling', () => {
    it('ajoute exactement maxNewTokens tokens', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 3,
        })
        model.outputBias[1] = 3

        const result = generateTokenIdsWithSampling(model, [0], {
            maxNewTokens: 4,
            strategy: 'greedy',
        })

        expect(result.generatedTokenIds).toHaveLength(4)
        expect(result.tokenIds).toHaveLength(5)
    })

    it('donne la même génération avec la même seed', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 3,
        })

        const firstResult = generateTokenIdsWithSampling(model, [0], {
            maxNewTokens: 5,
            seed: 123,
            strategy: 'temperature',
            temperature: 1.5,
        })
        const secondResult = generateTokenIdsWithSampling(model, [0], {
            maxNewTokens: 5,
            seed: 123,
            strategy: 'temperature',
            temperature: 1.5,
        })

        expect(firstResult.generatedTokenIds).toEqual(secondResult.generatedTokenIds)
    })

    it('peut donner des générations différentes avec des seeds différentes', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 3,
        })

        const firstResult = generateTokenIdsWithSampling(model, [0], {
            maxNewTokens: 8,
            seed: 1,
            strategy: 'temperature',
            temperature: 1.5,
        })
        const secondResult = generateTokenIdsWithSampling(model, [0], {
            maxNewTokens: 8,
            seed: 3_000_000_000,
            strategy: 'temperature',
            temperature: 1.5,
        })

        expect(firstResult.generatedTokenIds).not.toEqual(secondResult.generatedTokenIds)
    })
})

describe('generateTextWithSampling', () => {
    it('retourne un texte commençant par le prompt', () => {
        const tokenizer = createCharacterTokenizer('abc')
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: tokenizer.vocabularySize,
        })

        const result = generateTextWithSampling(model, tokenizer, 'ab', {
            maxNewTokens: 2,
            seed: 123,
            strategy: 'temperature',
            temperature: 1,
        })

        expect(result.text.startsWith('ab')).toBe(true)
    })

    it('fonctionne avec tokenizer, mini corpus, dataset loader, modèle entraîné et sampling', async () => {
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

        const result = generateTextWithSampling(model, tokenizer, 'bonj', {
            maxNewTokens: 3,
            seed: 123,
            strategy: 'topK',
            temperature: 1,
            topK: 3,
        })

        expect(result.text.startsWith('bonj')).toBe(true)
        expect(result.generatedTokenIds).toHaveLength(3)
        expect(result.steps).toHaveLength(3)
    })
})
