import { platform, release, type } from 'node:os'
import { performance } from 'node:perf_hooks'

import * as baseTf from '@tensorflow/tfjs'

export type RuntimeEnvironmentInfo = {
    readonly nodeVersion: string
    readonly platform: NodeJS.Platform
    readonly architecture: NodeJS.Architecture
    readonly osType: string
    readonly osRelease: string
    readonly currentBackend: string
    readonly tfjsVersion: string
}

export type TfjsNodeGpuBackendLoadResult =
    | {
          readonly available: true
          readonly backendName: string
          readonly tfjsVersion: string
          readonly tensorCount: number
      }
    | {
          readonly available: false
          readonly backendName: string
          readonly tfjsVersion: string
          readonly errorMessage: string
          readonly guidance: readonly string[]
      }

export type TfjsNodeGpuSmokeTestResult =
    | {
          readonly available: true
          readonly backendName: string
          readonly expected: readonly (readonly number[])[]
          readonly actual: readonly (readonly number[])[]
          readonly tensorCountBefore: number
          readonly tensorCountAfter: number
      }
    | {
          readonly available: false
          readonly expected: readonly (readonly number[])[]
          readonly errorMessage: string
          readonly guidance: readonly string[]
      }

export type TfjsNodeGpuBenchmarkOptions = {
    readonly matrixSize: number
    readonly iterations: number
}

export type TfjsNodeGpuBenchmarkResult =
    | {
          readonly available: true
          readonly backendName: string
          readonly matrixSize: number
          readonly iterations: number
          readonly durationMs: number
          readonly averageIterationMs: number
      }
    | {
          readonly available: false
          readonly matrixSize: number
          readonly iterations: number
          readonly errorMessage: string
          readonly guidance: readonly string[]
      }

type TfjsModule = typeof import('@tensorflow/tfjs')

const tfjsNodeGpuPackageName = '@tensorflow/tfjs-node-gpu'
const expectedSmokeTestResult = [
    [19, 22],
    [43, 50],
] as const

export function getRuntimeEnvironmentInfo(): RuntimeEnvironmentInfo {
    return {
        architecture: process.arch,
        currentBackend: getCurrentBaseBackendName(),
        nodeVersion: process.version,
        osRelease: release(),
        osType: type(),
        platform: platform(),
        tfjsVersion: baseTf.version.tfjs,
    }
}

export async function loadTfjsNodeGpuBackend(): Promise<TfjsNodeGpuBackendLoadResult> {
    try {
        const tf = await importTfjsNodeGpu()

        await tf.ready()

        return {
            available: true,
            backendName: tf.getBackend(),
            tensorCount: tf.memory().numTensors,
            tfjsVersion: tf.version.tfjs,
        }
    } catch (error) {
        return createUnavailableBackendResult(error)
    }
}

export async function runTfjsNodeGpuSmokeTest(): Promise<TfjsNodeGpuSmokeTestResult> {
    const backend = await loadTfjsNodeGpuBackend()

    if (!backend.available) {
        return {
            available: false,
            errorMessage: backend.errorMessage,
            expected: expectedSmokeTestResult,
            guidance: backend.guidance,
        }
    }

    const tf = await importTfjsNodeGpu()
    const tensorCountBefore = tf.memory().numTensors
    const actual = tf.tidy(() => {
        const left = tf.tensor2d(
            [
                [1, 2],
                [3, 4],
            ],
            [2, 2],
        )
        const right = tf.tensor2d(
            [
                [5, 6],
                [7, 8],
            ],
            [2, 2],
        )

        return tf.matMul(left, right).arraySync() as number[][]
    })
    const tensorCountAfter = tf.memory().numTensors

    return {
        actual,
        available: true,
        backendName: backend.backendName,
        expected: expectedSmokeTestResult,
        tensorCountAfter,
        tensorCountBefore,
    }
}

export async function benchmarkTfjsNodeGpuMatMul(
    options: TfjsNodeGpuBenchmarkOptions,
): Promise<TfjsNodeGpuBenchmarkResult> {
    validatePositiveInteger(options.matrixSize, 'matrixSize')
    validatePositiveInteger(options.iterations, 'iterations')

    const backend = await loadTfjsNodeGpuBackend()

    if (!backend.available) {
        return {
            available: false,
            errorMessage: backend.errorMessage,
            guidance: backend.guidance,
            iterations: options.iterations,
            matrixSize: options.matrixSize,
        }
    }

    const tf = await importTfjsNodeGpu()
    const durationMs = await runBenchmark(tf, options)

    return {
        available: true,
        averageIterationMs: durationMs / options.iterations,
        backendName: backend.backendName,
        durationMs,
        iterations: options.iterations,
        matrixSize: options.matrixSize,
    }
}

async function importTfjsNodeGpu(): Promise<TfjsModule> {
    const importedModule = (await import(tfjsNodeGpuPackageName)) as TfjsModule

    return importedModule
}

async function runBenchmark(tf: TfjsModule, options: TfjsNodeGpuBenchmarkOptions): Promise<number> {
    const left = tf.randomNormal([options.matrixSize, options.matrixSize])
    const right = tf.randomNormal([options.matrixSize, options.matrixSize])
    const startedAt = performance.now()

    try {
        for (let iteration = 0; iteration < options.iterations; iteration++) {
            const result = tf.matMul(left, right)

            // data() force le calcul à se terminer. Sans cela, on mesurerait surtout
            // la planification de l'opération, pas son exécution réelle.
            await result.data()
            result.dispose()
        }

        return performance.now() - startedAt
    } finally {
        left.dispose()
        right.dispose()
    }
}

function createUnavailableBackendResult(error: unknown): TfjsNodeGpuBackendLoadResult {
    const errorMessage =
        error instanceof Error
            ? error.message
            : `Erreur inconnue pendant le chargement de ${tfjsNodeGpuPackageName}.`

    return {
        available: false,
        backendName: getCurrentBaseBackendName(),
        errorMessage,
        guidance: [
            `${tfjsNodeGpuPackageName} doit être installé dans l'environnement Node.js courant.`,
            'Le backend GPU TensorFlow.js Node repose sur CUDA et vise Linux.',
            'Sous Windows, utilise WSL2 avec une distribution Ubuntu et vérifie que nvidia-smi fonctionne dans WSL.',
            'Sur macOS, ce backend GPU CUDA n’est pas adapté; garde @tensorflow/tfjs ou teste @tensorflow/tfjs-node CPU.',
        ],
        tfjsVersion: baseTf.version.tfjs,
    }
}

function getCurrentBaseBackendName(): string {
    const backendName = baseTf.getBackend() as unknown

    if (typeof backendName === 'string' && backendName.length > 0) {
        return backendName
    }

    return 'non initialisé'
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
            `${name} doit être un entier strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}
