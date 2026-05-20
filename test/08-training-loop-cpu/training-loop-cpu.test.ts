import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import {
    createNextTokenExamples,
    createTrainableTokenBiasModel,
    crossEntropyLoss,
    perplexityFromLoss,
    softmax,
    trainNextTokenModel,
} from '../../src/modules/08-training-loop-cpu/index.js'

describe('createNextTokenExamples', () => {
    it('crée les bons couples contexte/cible', () => {
        const examples = createNextTokenExamples([10, 11, 12, 13], { contextLength: 2 })

        expect(examples).toEqual([
            {
                inputTokenIds: [10, 11],
                targetTokenId: 12,
            },
            {
                inputTokenIds: [11, 12],
                targetTokenId: 13,
            },
        ])
    })

    it('rejette un contextLength invalide', () => {
        expect(() => createNextTokenExamples([1, 2, 3], { contextLength: 0 })).toThrow(
            'contextLength doit être un entier strictement positif.',
        )
    })

    it('retourne une liste vide si la séquence est trop courte', () => {
        expect(createNextTokenExamples([1, 2], { contextLength: 2 })).toEqual([])
    })
})

describe('softmax', () => {
    it('retourne des probabilités qui somment à 1', () => {
        const probabilities = softmax([1, 2, 3])
        const sum = probabilities.reduce((total, probability) => total + probability, 0)

        expect(sum).toBeCloseTo(1)
    })

    it('favorise les logits plus élevés', () => {
        const probabilities = softmax([0, 2])

        expect(probabilities[1]).toBeGreaterThan(probabilities[0] ?? 0)
    })
})

describe('crossEntropyLoss', () => {
    it('est faible quand la bonne cible a une probabilité élevée', () => {
        expect(crossEntropyLoss([0.95, 0.05], 0)).toBeLessThan(0.1)
    })

    it('rejette un target id hors vocabulaire', () => {
        expect(() => crossEntropyLoss([0.5, 0.5], 2)).toThrow(
            'targetTokenId doit être un entier entre 0 et 1. Valeur reçue: 2.',
        )
    })
})

describe('perplexityFromLoss', () => {
    it('retourne Math.exp(loss)', () => {
        expect(perplexityFromLoss(2)).toBe(Math.exp(2))
    })
})

describe('createTrainableTokenBiasModel', () => {
    it('valide vocabularySize', () => {
        expect(() => createTrainableTokenBiasModel({ vocabularySize: 0 })).toThrow(
            'vocabularySize doit être un entier strictement positif.',
        )
    })
})

describe('trainNextTokenModel', () => {
    it('réduit la loss sur des exemples répétitifs', () => {
        const model = createTrainableTokenBiasModel({ vocabularySize: 2 })
        const examples = [
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [1], targetTokenId: 1 },
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [1], targetTokenId: 1 },
        ]

        const history = trainNextTokenModel(model, examples, {
            epochs: 10,
            learningRate: 0.5,
        })

        expect(history.finalLoss).toBeLessThan(history.initialLoss)
    })

    it('retourne un historique avec une métrique par epoch', () => {
        const model = createTrainableTokenBiasModel({ vocabularySize: 2 })
        const examples = [
            { inputTokenIds: [0], targetTokenId: 1 },
            { inputTokenIds: [1], targetTokenId: 0 },
        ]

        const history = trainNextTokenModel(model, examples, {
            epochs: 3,
            learningRate: 0.1,
        })

        expect(history.epochs).toHaveLength(3)
        expect(history.epochs[0]?.epoch).toBe(1)
        expect(history.epochs[2]?.epoch).toBe(3)
    })

    it('fonctionne avec le tokenizer, le mini corpus et le dataset loader', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength: 4 })
        const model = createTrainableTokenBiasModel({ vocabularySize: tokenizer.vocabularySize })

        const history = trainNextTokenModel(model, examples.slice(0, 20), {
            epochs: 2,
            learningRate: 0.2,
        })

        expect(examples.length).toBeGreaterThan(0)
        expect(history.epochs).toHaveLength(2)
        expect(history.finalLoss).toBeGreaterThanOrEqual(0)
    })
})
