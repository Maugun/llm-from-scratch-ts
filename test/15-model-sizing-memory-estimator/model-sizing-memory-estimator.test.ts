import { describe, expect, it } from 'vitest'

import {
    compareModelSizes,
    estimateMiniTransformerSize,
    formatBytes,
    formatParameterCount,
} from '../../src/modules/15-model-sizing-memory-estimator/index.js'

describe('estimateMiniTransformerSize', () => {
    it('calcule correctement les paramètres d’un modèle tiny', () => {
        const estimate = estimateMiniTransformerSize({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 1,
            vocabularySize: 10,
        })

        expect(estimate.parameters.tokenEmbeddings).toBe(80)
        expect(estimate.parameters.positionEmbeddings).toBe(32)
        expect(estimate.parameters.attentionPerLayer).toBe(256)
        expect(estimate.parameters.feedForwardPerLayer).toBe(280)
        expect(estimate.parameters.transformerBlockPerLayer).toBe(536)
        expect(estimate.parameters.outputProjection).toBe(90)
        expect(estimate.parameters.total).toBe(738)
    })

    it('utilise float32 et batchSize 1 par défaut', () => {
        const estimate = estimateMiniTransformerSize({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 1,
            vocabularySize: 10,
        })

        expect(estimate.config.bytesPerParameter).toBe(4)
        expect(estimate.config.batchSize).toBe(1)
    })

    it('calcule le coût attention avec le contexte, le batch et les layers', () => {
        const estimate = estimateMiniTransformerSize({
            batchSize: 2,
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 3,
            vocabularySize: 10,
        })

        expect(estimate.attention.scoresPerLayer).toBe(32)
        expect(estimate.attention.scoresTotal).toBe(96)
    })

    it('estime plus de mémoire pour l’entraînement que pour les paramètres seuls', () => {
        const estimate = estimateMiniTransformerSize({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 1,
            vocabularySize: 10,
        })

        expect(estimate.memory.trainingParameterBytes).toBeGreaterThan(
            estimate.memory.parameterBytes,
        )
    })

    it('augmente le nombre de paramètres quand layerCount augmente', () => {
        const oneLayer = estimateMiniTransformerSize({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 1,
            vocabularySize: 10,
        })
        const threeLayers = estimateMiniTransformerSize({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 3,
            vocabularySize: 10,
        })

        expect(threeLayers.parameters.total).toBeGreaterThan(oneLayer.parameters.total)
    })

    it('augmente le coût attention quand contextLength augmente', () => {
        const shortContext = estimateMiniTransformerSize({
            contextLength: 4,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 1,
            vocabularySize: 10,
        })
        const longContext = estimateMiniTransformerSize({
            contextLength: 8,
            embeddingDimension: 8,
            feedForwardDimension: 16,
            layerCount: 1,
            vocabularySize: 10,
        })

        expect(longContext.attention.scoresTotal).toBeGreaterThan(
            shortContext.attention.scoresTotal,
        )
    })

    it('rejette les dimensions invalides', () => {
        expect(() =>
            estimateMiniTransformerSize({
                contextLength: 4,
                embeddingDimension: 8,
                feedForwardDimension: 16,
                layerCount: 1,
                vocabularySize: 0,
            }),
        ).toThrow('vocabularySize doit être un entier strictement positif.')
        expect(() =>
            estimateMiniTransformerSize({
                contextLength: 0,
                embeddingDimension: 8,
                feedForwardDimension: 16,
                layerCount: 1,
                vocabularySize: 10,
            }),
        ).toThrow('contextLength doit être un entier strictement positif.')
        expect(() =>
            estimateMiniTransformerSize({
                contextLength: 4,
                embeddingDimension: 0,
                feedForwardDimension: 16,
                layerCount: 1,
                vocabularySize: 10,
            }),
        ).toThrow('embeddingDimension doit être un entier strictement positif.')
        expect(() =>
            estimateMiniTransformerSize({
                contextLength: 4,
                embeddingDimension: 8,
                feedForwardDimension: 0,
                layerCount: 1,
                vocabularySize: 10,
            }),
        ).toThrow('feedForwardDimension doit être un entier strictement positif.')
        expect(() =>
            estimateMiniTransformerSize({
                contextLength: 4,
                embeddingDimension: 8,
                feedForwardDimension: 16,
                layerCount: 0,
                vocabularySize: 10,
            }),
        ).toThrow('layerCount doit être un entier strictement positif.')
    })
})

describe('compareModelSizes', () => {
    it('retourne une estimation par configuration', () => {
        const estimates = compareModelSizes([
            {
                contextLength: 4,
                embeddingDimension: 8,
                feedForwardDimension: 16,
                layerCount: 1,
                name: 'tiny',
                vocabularySize: 10,
            },
            {
                contextLength: 8,
                embeddingDimension: 16,
                feedForwardDimension: 64,
                layerCount: 2,
                name: 'small',
                vocabularySize: 20,
            },
        ])

        expect(estimates).toHaveLength(2)
        expect(estimates[0]?.name).toBe('tiny')
        expect(estimates[1]?.name).toBe('small')
    })
})

describe('formatBytes', () => {
    it('formate les bytes et les kibibytes', () => {
        expect(formatBytes(512)).toBe('512 B')
        expect(formatBytes(2048)).toBe('2.00 KB')
    })

    it('rejette une taille négative', () => {
        expect(() => formatBytes(-1)).toThrow('bytes doit être un nombre fini positif ou nul.')
    })
})

describe('formatParameterCount', () => {
    it('formate les paramètres en unités lisibles', () => {
        expect(formatParameterCount(999)).toBe('999')
        expect(formatParameterCount(1_500)).toBe('1.50K')
        expect(formatParameterCount(2_000_000)).toBe('2.00M')
    })

    it('rejette un nombre de paramètres négatif', () => {
        expect(() => formatParameterCount(-1)).toThrow(
            'parameterCount doit être un nombre fini positif ou nul.',
        )
    })
})
