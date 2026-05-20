import { describe, expect, it } from 'vitest'

import {
    benchmarkTfjsNodeGpuMatMul,
    getRuntimeEnvironmentInfo,
    loadTfjsNodeGpuBackend,
    runTfjsNodeGpuSmokeTest,
} from '../../src/modules/16-tfjs-node-gpu-backend/index.js'

describe('getRuntimeEnvironmentInfo', () => {
    it('retourne les informations runtime principales', () => {
        const info = getRuntimeEnvironmentInfo()

        expect(info.nodeVersion).toMatch(/^v\d+/)
        expect(info.platform.length).toBeGreaterThan(0)
        expect(info.architecture.length).toBeGreaterThan(0)
        expect(info.tfjsVersion.length).toBeGreaterThan(0)
    })
})

describe('loadTfjsNodeGpuBackend', () => {
    it('retourne un résultat pédagogique même si le backend GPU est indisponible', async () => {
        const result = await loadTfjsNodeGpuBackend()

        if (result.available) {
            expect(result.backendName.length).toBeGreaterThan(0)
            expect(result.tfjsVersion.length).toBeGreaterThan(0)
            expect(result.tensorCount).toBeGreaterThanOrEqual(0)
        } else {
            expect(result.errorMessage.length).toBeGreaterThan(0)
            expect(result.guidance.length).toBeGreaterThan(0)
        }
    })
})

describe('runTfjsNodeGpuSmokeTest', () => {
    it('retourne le bon résultat ou un skip pédagogique', async () => {
        const result = await runTfjsNodeGpuSmokeTest()

        expect(result.expected).toEqual([
            [19, 22],
            [43, 50],
        ])

        if (result.available) {
            expect(result.actual).toEqual(result.expected)
            expect(result.tensorCountAfter).toBe(result.tensorCountBefore)
        } else {
            expect(result.errorMessage.length).toBeGreaterThan(0)
            expect(result.guidance.length).toBeGreaterThan(0)
        }
    })
})

describe('benchmarkTfjsNodeGpuMatMul', () => {
    it('rejette matrixSize <= 0', async () => {
        await expect(
            benchmarkTfjsNodeGpuMatMul({
                iterations: 1,
                matrixSize: 0,
            }),
        ).rejects.toThrow('matrixSize doit être un entier strictement positif.')
    })

    it('rejette iterations <= 0', async () => {
        await expect(
            benchmarkTfjsNodeGpuMatMul({
                iterations: 0,
                matrixSize: 2,
            }),
        ).rejects.toThrow('iterations doit être un entier strictement positif.')
    })

    it('retourne une durée finie ou un skip pédagogique', async () => {
        const result = await benchmarkTfjsNodeGpuMatMul({
            iterations: 1,
            matrixSize: 2,
        })

        if (result.available) {
            expect(Number.isFinite(result.durationMs)).toBe(true)
            expect(result.durationMs).toBeGreaterThanOrEqual(0)
            expect(result.averageIterationMs).toBeGreaterThanOrEqual(0)
        } else {
            expect(result.errorMessage.length).toBeGreaterThan(0)
            expect(result.guidance.length).toBeGreaterThan(0)
        }
    })
})
