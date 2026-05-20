import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import { createNextTokenExamples } from '../../src/modules/08-training-loop-cpu/index.js'
import {
    computeTfjsNextTokenAverageLoss,
    createTfjsNextTokenModel,
    disposeTfjsNextTokenModel,
    predictMostLikelyNextToken,
    predictNextTokenLogits,
    predictNextTokenProbabilities,
    trainTfjsNextTokenModel,
    type TfjsNextTokenModel,
} from '../../src/modules/13-tfjs-next-token-model/index.js'

const modelsToDispose: TfjsNextTokenModel[] = []

afterEach(() => {
    for (const model of modelsToDispose.splice(0)) {
        disposeTfjsNextTokenModel(model)
    }
})

describe('createTfjsNextTokenModel', () => {
    it('crée les variables aux bonnes shapes', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 4,
                embeddingDimension: 8,
                vocabularySize: 10,
            }),
        )

        expect(model.tokenEmbeddings.shape).toEqual([10, 8])
        expect(model.positionEmbeddings.shape).toEqual([4, 8])
        expect(model.outputWeights.shape).toEqual([32, 10])
        expect(model.outputBias.shape).toEqual([10])
    })

    it('rejette vocabularySize <= 0', () => {
        expect(() =>
            createTfjsNextTokenModel({
                contextLength: 4,
                embeddingDimension: 8,
                vocabularySize: 0,
            }),
        ).toThrow('vocabularySize doit être un entier strictement positif.')
    })

    it('rejette contextLength <= 0', () => {
        expect(() =>
            createTfjsNextTokenModel({
                contextLength: 0,
                embeddingDimension: 8,
                vocabularySize: 10,
            }),
        ).toThrow('contextLength doit être un entier strictement positif.')
    })

    it('rejette embeddingDimension <= 0', () => {
        expect(() =>
            createTfjsNextTokenModel({
                contextLength: 4,
                embeddingDimension: 0,
                vocabularySize: 10,
            }),
        ).toThrow('embeddingDimension doit être un entier strictement positif.')
    })
})

describe('predictNextTokenLogits', () => {
    it('retourne un logit par token du vocabulaire', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 3,
                vocabularySize: 5,
            }),
        )
        const logits = predictNextTokenLogits(model, [0, 1])

        expect(logits.shape).toEqual([5])

        logits.dispose()
    })

    it("rejette un contexte qui n'a pas contextLength tokens", () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 3,
                vocabularySize: 5,
            }),
        )

        expect(() => predictNextTokenLogits(model, [0])).toThrow(
            'Le contexte doit contenir 2 tokens.',
        )
    })

    it('rejette un token de contexte hors vocabulaire', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 3,
                vocabularySize: 5,
            }),
        )

        expect(() => predictNextTokenLogits(model, [0, 5])).toThrow(
            'inputTokenIds[1] doit être un entier entre 0 et 4.',
        )
    })
})

describe('predictNextTokenProbabilities', () => {
    it('retourne une distribution normalisée', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 3,
                vocabularySize: 5,
            }),
        )
        const probabilities = predictNextTokenProbabilities(model, [0, 1])
        const sum = probabilities.reduce((total, probability) => total + probability, 0)

        expect(probabilities).toHaveLength(5)
        expect(sum).toBeCloseTo(1)
    })
})

describe('predictMostLikelyNextToken', () => {
    it('retourne un id valide', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 3,
                vocabularySize: 5,
            }),
        )
        const tokenId = predictMostLikelyNextToken(model, [0, 1])

        expect(tokenId).toBeGreaterThanOrEqual(0)
        expect(tokenId).toBeLessThan(5)
    })
})

describe('computeTfjsNextTokenAverageLoss', () => {
    it('retourne une loss positive', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 3,
                vocabularySize: 4,
            }),
        )
        const loss = computeTfjsNextTokenAverageLoss(model, [
            { inputTokenIds: [0, 1], targetTokenId: 2 },
            { inputTokenIds: [1, 2], targetTokenId: 3 },
        ])

        expect(loss).toBeGreaterThan(0)
    })
})

describe('trainTfjsNextTokenModel', () => {
    it('réduit la loss sur un dataset répétitif simple', () => {
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 2,
                embeddingDimension: 4,
                seed: 42,
                vocabularySize: 3,
            }),
        )
        const examples = [
            { inputTokenIds: [0, 1], targetTokenId: 2 },
            { inputTokenIds: [0, 1], targetTokenId: 2 },
            { inputTokenIds: [0, 1], targetTokenId: 2 },
            { inputTokenIds: [1, 0], targetTokenId: 1 },
        ]

        const history = trainTfjsNextTokenModel(model, examples, {
            epochs: 25,
            learningRate: 0.2,
        })

        expect(history.finalLoss).toBeLessThan(history.initialLoss)
    })

    it('fonctionne avec tokenizer, mini corpus, dataset loader et exemples du module 8', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength: 4 })
        const model = trackModel(
            createTfjsNextTokenModel({
                contextLength: 4,
                embeddingDimension: 8,
                seed: 13,
                vocabularySize: tokenizer.vocabularySize,
            }),
        )

        const history = trainTfjsNextTokenModel(model, examples, {
            epochs: 2,
            learningRate: 0.1,
        })

        expect(history.epochs).toHaveLength(2)
        expect(history.finalLoss).toBeGreaterThan(0)
    })
})

describe('disposeTfjsNextTokenModel', () => {
    it('peut être appelé sans erreur', () => {
        const model = createTfjsNextTokenModel({
            contextLength: 2,
            embeddingDimension: 3,
            vocabularySize: 5,
        })

        expect(() => {
            disposeTfjsNextTokenModel(model)
        }).not.toThrow()
    })
})

function trackModel(model: TfjsNextTokenModel): TfjsNextTokenModel {
    modelsToDispose.push(model)

    return model
}
