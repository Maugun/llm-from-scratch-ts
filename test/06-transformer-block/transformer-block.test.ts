import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import { createEmbeddingTable } from '../../src/modules/04-embeddings/index.js'
import {
    addVectors,
    applyFeedForward,
    createTransformerBlock,
    layerNormalize,
} from '../../src/modules/06-transformer-block/index.js'

describe('addVectors', () => {
    it('additionne deux vecteurs de même dimension', () => {
        expect(addVectors([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9])
    })

    it('rejette deux dimensions incompatibles', () => {
        expect(() => addVectors([1, 2], [1])).toThrow(
            'Les deux vecteurs doivent avoir la même dimension.',
        )
    })
})

describe('layerNormalize', () => {
    it('garde la même dimension', () => {
        expect(layerNormalize([1, 2, 3])).toHaveLength(3)
    })

    it('centre approximativement les valeurs autour de 0', () => {
        const normalizedVector = layerNormalize([1, 2, 3])
        const mean =
            normalizedVector.reduce((sum, value) => sum + value, 0) / normalizedVector.length

        expect(mean).toBeCloseTo(0)
    })

    it('ramène la variance proche de 1', () => {
        const normalizedVector = layerNormalize([1, 2, 3])
        const mean =
            normalizedVector.reduce((sum, value) => sum + value, 0) / normalizedVector.length
        const variance =
            normalizedVector.reduce((sum, value) => {
                const distanceFromMean = value - mean

                return sum + distanceFromMean * distanceFromMean
            }, 0) / normalizedVector.length

        expect(variance).toBeCloseTo(1, 4)
    })

    it('rejette un vecteur vide', () => {
        expect(() => layerNormalize([])).toThrow('layerNormalize attend au moins une valeur.')
    })
})

describe('applyFeedForward', () => {
    it('retourne un vecteur de dimension embeddingDimension', () => {
        const outputVector = applyFeedForward([1, -1], {
            inputWeights: [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            outputWeights: [
                [1, 0, 0],
                [0, 1, 1],
            ],
        })

        expect(outputVector).toHaveLength(2)
    })

    it('rejette des poids qui ne reviennent pas à embeddingDimension', () => {
        expect(() =>
            applyFeedForward([1, -1], {
                inputWeights: [
                    [1, 0],
                    [0, 1],
                ],
                outputWeights: [[1, 0]],
            }),
        ).toThrow('outputWeights doit produire des vecteurs de 2 dimensions. Dimension reçue: 1.')
    })
})

describe('createTransformerBlock', () => {
    it('initialise les poids déterministiquement avec le même seed', () => {
        const blockA = createTransformerBlock({
            embeddingDimension: 2,
            feedForwardDimension: 4,
            seed: 123,
        })
        const blockB = createTransformerBlock({
            embeddingDimension: 2,
            feedForwardDimension: 4,
            seed: 123,
        })

        expect(blockA.attentionOutputWeights).toEqual(blockB.attentionOutputWeights)
        expect(blockA.feedForwardWeights).toEqual(blockB.feedForwardWeights)
    })

    it('produit des poids différents avec des seeds différents', () => {
        const blockA = createTransformerBlock({
            embeddingDimension: 2,
            feedForwardDimension: 4,
            seed: 123,
        })
        const blockB = createTransformerBlock({
            embeddingDimension: 2,
            feedForwardDimension: 4,
            seed: 456,
        })

        expect(blockA.feedForwardWeights).not.toEqual(blockB.feedForwardWeights)
    })

    it('retourne une sortie par position', () => {
        const block = createTransformerBlock({
            embeddingDimension: 2,
            feedForwardDimension: 4,
            seed: 123,
        })
        const result = block.applyTransformerBlock([
            [1, 0],
            [0, 1],
            [1, 1],
        ])

        expect(result.outputVectors).toHaveLength(3)
        expect(result.attentionResidualVectors).toHaveLength(3)
    })

    it('garde la dimension embeddingDimension pour chaque sortie', () => {
        const block = createTransformerBlock({
            attentionDimension: 3,
            embeddingDimension: 2,
            feedForwardDimension: 4,
            seed: 123,
        })
        const result = block.applyTransformerBlock([
            [1, 0],
            [0, 1],
        ])

        expect(result.outputVectors[0]).toHaveLength(2)
        expect(result.outputVectors[1]).toHaveLength(2)
    })

    it('rejette les dimensions invalides', () => {
        expect(() => createTransformerBlock({ embeddingDimension: 0 })).toThrow(
            'embeddingDimension doit être un entier strictement positif.',
        )
        expect(() =>
            createTransformerBlock({ attentionDimension: 0, embeddingDimension: 2 }),
        ).toThrow('attentionDimension doit être un entier strictement positif.')
        expect(() =>
            createTransformerBlock({ embeddingDimension: 2, feedForwardDimension: 0 }),
        ).toThrow('feedForwardDimension doit être un entier strictement positif.')
    })

    it("rejette les vecteurs d'entrée incompatibles", () => {
        const block = createTransformerBlock({
            embeddingDimension: 2,
        })

        expect(() => block.applyTransformerBlock([])).toThrow(
            'applyTransformerBlock attend au moins un vecteur.',
        )
        expect(() => block.applyTransformerBlock([[1, 2, 3]])).toThrow(
            "Le vecteur d'entrée 0 doit avoir 2 dimensions. Dimension reçue: 3.",
        )
    })

    it('fonctionne avec le tokenizer, le mini corpus et les embeddings', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const embeddingTable = createEmbeddingTable({
            embeddingDimension: 4,
            vocabularySize: tokenizer.vocabularySize,
        })
        const block = createTransformerBlock({
            embeddingDimension: embeddingTable.embeddingDimension,
            feedForwardDimension: 8,
        })
        const inputVectors = embeddingTable.embedSequence(dataset.trainTokenIds.slice(0, 3))
        const result = block.applyTransformerBlock(inputVectors)

        expect(result.outputVectors).toHaveLength(inputVectors.length)
        expect(result.outputVectors[0]).toHaveLength(embeddingTable.embeddingDimension)
    })
})
