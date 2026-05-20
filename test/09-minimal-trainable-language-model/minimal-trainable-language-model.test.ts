import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import { createNextTokenExamples } from '../../src/modules/08-training-loop-cpu/index.js'
import {
    computeAverageLoss,
    createMinimalLanguageModel,
    predictMostLikelyNextToken,
    predictNextTokenProbabilities,
    trainMinimalLanguageModel,
} from '../../src/modules/09-minimal-trainable-language-model/index.js'

describe('createMinimalLanguageModel', () => {
    it('crée un modèle aux bonnes dimensions', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })

        expect(model.outputBias).toHaveLength(3)
        expect(model.contextWeights).toHaveLength(2)
        expect(model.contextWeights[0]).toHaveLength(3)
        expect(model.contextWeights[0]?.[0]).toHaveLength(3)
    })

    it('valide vocabularySize et contextLength', () => {
        expect(() =>
            createMinimalLanguageModel({
                contextLength: 1,
                vocabularySize: 0,
            }),
        ).toThrow('vocabularySize doit être un entier strictement positif.')

        expect(() =>
            createMinimalLanguageModel({
                contextLength: 0,
                vocabularySize: 2,
            }),
        ).toThrow('contextLength doit être un entier strictement positif.')
    })
})

describe('predictNextTokenProbabilities', () => {
    it('retourne une distribution qui somme à 1', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })
        const probabilities = predictNextTokenProbabilities(model, [0, 1])
        const sum = probabilities.reduce((total, probability) => total + probability, 0)

        expect(sum).toBeCloseTo(1)
    })

    it('rejette un contexte de mauvaise longueur', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })

        expect(() => predictNextTokenProbabilities(model, [0])).toThrow(
            'Le contexte doit contenir 2 tokens. Nombre reçu: 1.',
        )
    })

    it('rejette un token de contexte hors vocabulaire', () => {
        const model = createMinimalLanguageModel({
            contextLength: 2,
            vocabularySize: 3,
        })

        expect(() => predictNextTokenProbabilities(model, [0, 3])).toThrow(
            'inputTokenIds[1] doit être un entier entre 0 et 2. Valeur reçue: 3.',
        )
    })
})

describe('predictMostLikelyNextToken', () => {
    it('retourne le token le plus probable', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 3,
        })
        model.outputBias[2] = 4

        expect(predictMostLikelyNextToken(model, [0])).toBe(2)
    })
})

describe('computeAverageLoss', () => {
    it('retourne une loss positive', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 2,
        })

        expect(
            computeAverageLoss(model, [{ inputTokenIds: [0], targetTokenId: 1 }]),
        ).toBeGreaterThan(0)
    })
})

describe('trainMinimalLanguageModel', () => {
    it('réduit la loss sur un dataset répétitif', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 3,
        })
        const examples = [
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [1], targetTokenId: 2 },
            { inputTokenIds: [1], targetTokenId: 2 },
        ]

        const history = trainMinimalLanguageModel(model, examples, {
            epochs: 20,
            learningRate: 0.3,
        })

        expect(history.finalLoss).toBeLessThan(history.initialLoss)
    })

    it('produit des distributions différentes pour deux contextes après entraînement', () => {
        const model = createMinimalLanguageModel({
            contextLength: 1,
            vocabularySize: 3,
        })
        const examples = [
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [1], targetTokenId: 2 },
            { inputTokenIds: [1], targetTokenId: 2 },
        ]

        trainMinimalLanguageModel(model, examples, {
            epochs: 30,
            learningRate: 0.3,
        })

        expect(predictMostLikelyNextToken(model, [0])).toBe(1)
        expect(predictMostLikelyNextToken(model, [1])).toBe(2)
        expect(predictNextTokenProbabilities(model, [0])).not.toEqual(
            predictNextTokenProbabilities(model, [1]),
        )
    })

    it('fonctionne avec le tokenizer, le mini corpus, le dataset loader et les exemples du module 8', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength: 4 })
        const model = createMinimalLanguageModel({
            contextLength: 4,
            vocabularySize: tokenizer.vocabularySize,
        })

        const history = trainMinimalLanguageModel(model, examples.slice(0, 40), {
            epochs: 2,
            learningRate: 0.2,
        })

        expect(examples.length).toBeGreaterThan(0)
        expect(history.epochs).toHaveLength(2)
        expect(history.finalLoss).toBeGreaterThanOrEqual(0)
    })
})
