import { existsSync } from 'node:fs'
import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { formatBytes } from '../15-model-sizing-memory-estimator/index.js'
import {
    createLongCorpusPipeline,
    disposeTensorNextTokenBatch,
    iterateNextTokenBatches,
    loadLongCorpusText,
    nextTokenBatchToTensors,
    savePreparedLongCorpusDataset,
    type LongCorpusPipeline,
    type NextTokenBatch,
} from './index.js'

const privateCorpusPath = join(process.cwd(), 'data', 'private', 'long-corpus.txt')
const fallbackCorpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const cachePath = join(process.cwd(), 'data', 'cache', 'long-corpus.dataset.json')
const usesPrivateCorpus = existsSync(privateCorpusPath)
const corpusPath = usesPrivateCorpus ? privateCorpusPath : fallbackCorpusPath
const contextLength = usesPrivateCorpus ? 128 : 8
const batchSize = usesPrivateCorpus ? 16 : 4
const validationRatio = usesPrivateCorpus ? 0.05 : 0.1

console.info('Module 17 - Pipeline long corpus')
console.info('')
console.info('But du module:')
console.info('  Préparer un corpus texte pour un entraînement next-token batché.')
console.info('  On ne lance pas encore de modèle: on prépare le carburant du module 18.')
console.info('')
console.info('Pipeline:')
console.info('1. Lire un fichier texte UTF-8')
console.info('2. Construire un tokenizer caractère')
console.info('3. Encoder le texte en token ids')
console.info('4. Séparer train / validation')
console.info('5. Générer les batches à la demande')
console.info('6. Convertir un batch en tenseurs TensorFlow.js')
console.info('')

if (!usesPrivateCorpus) {
    console.info('Aucun corpus privé trouvé dans data/private/long-corpus.txt.')
    console.info('La démo utilise data/tiny-corpus.txt pour rester exécutable partout.')
    console.info('Pour tester un vrai corpus local, crée data/private/long-corpus.txt.')
    console.info('')
}

const corpus = await loadLongCorpusText(corpusPath)
const tokenizer = createCharacterTokenizer(corpus.rawText)
let currentContextLength = contextLength
let pipeline = createPipeline(currentContextLength)
const firstBatch = getBatchByIndex(pipeline, 0)

console.info(`Corpus utilisé: ${corpus.filePath}`)
console.info(`Taille fichier: ${formatBytes(corpus.stats.byteLength)}`)
console.info(`Caractères: ${String(corpus.stats.characterCount)}`)
console.info(`Lignes: ${String(corpus.stats.lineCount)}`)
printPipelineStats(pipeline)
console.info('')

if (firstBatch === undefined) {
    console.info('Le corpus est trop court pour produire un batch avec ce contextLength.')
} else {
    console.info('Conversion TensorFlow.js:')
    console.info('  La démo convertit seulement le premier batch en tenseurs.')
    console.info(
        '  TensorFlow.js peut afficher un avertissement de backend Node; ce n’est pas bloquant.',
    )
    console.info('')

    const tensorBatch = nextTokenBatchToTensors(firstBatch)

    console.info('Premier batch:')
    printBatchSummary(firstBatch, pipeline)
    console.info(`  tensor inputTokenIds shape: [${tensorBatch.inputTokenIds.shape.join(', ')}]`)
    console.info(`  tensor targetTokenIds shape: [${tensorBatch.targetTokenIds.shape.join(', ')}]`)

    printFirstDecodedExample(firstBatch)
    disposeTensorNextTokenBatch(tensorBatch)
}

console.info('')
console.info('Cache dataset:')
console.info(`  Chemin: ${cachePath}`)
console.info('  La démo sauvegarde un cache JSON pédagogique.')
console.info('  Ce format est lisible, mais pas optimal pour de très gros corpus.')

await savePreparedLongCorpusDataset(pipeline, cachePath, {
    sourceFilePath: corpus.filePath,
})

console.info('  Cache écrit.')
console.info('')
console.info('À retenir:')
console.info('  1. On garde le texte long hors Git, dans data/private/.')
console.info('  2. On ne matérialise pas tous les exemples next-token.')
console.info('  3. Les batches sont produits à la demande.')
console.info('  4. La conversion TensorFlow.js prépare directement le module 18.')

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        'Mode non interactif détecté: lance cette démo dans un terminal pour inspecter des exemples et batches.',
    )
}

function createPipeline(nextContextLength: number): LongCorpusPipeline {
    return createLongCorpusPipeline(corpus.rawText, tokenizer, {
        batchSize,
        contextLength: nextContextLength,
        validationRatio,
    })
}

function printPipelineStats(nextPipeline: LongCorpusPipeline): void {
    console.info(`Vocabulaire: ${String(nextPipeline.vocabularySize)} caractères`)
    console.info(`Tokens totaux: ${String(nextPipeline.totalTokens)}`)
    console.info(`Tokens train: ${String(nextPipeline.trainTokenCount)}`)
    console.info(`Tokens validation: ${String(nextPipeline.validationTokenCount)}`)
    console.info(`Context length: ${String(nextPipeline.contextLength)}`)
    console.info(`Batch size: ${String(nextPipeline.batchSize)}`)
    console.info(`Exemples train estimés: ${String(nextPipeline.trainExampleCount)}`)
    console.info(`Batches train estimés: ${String(nextPipeline.trainBatchCount)}`)
}

function printBatchSummary(batch: NextTokenBatch, nextPipeline: LongCorpusPipeline): void {
    const estimatedBytes = estimateBatchBytes(batch, nextPipeline.contextLength)

    console.info(
        `  inputTokenIds: [${String(batch.inputTokenIds.length)}, ${String(
            nextPipeline.contextLength,
        )}]`,
    )
    console.info(`  targetTokenIds: [${String(batch.targetTokenIds.length)}]`)
    console.info(`  mémoire int32 approximative: ${formatBytes(estimatedBytes)}`)
}

function printFirstDecodedExample(batch: NextTokenBatch): void {
    const firstContext = batch.inputTokenIds[0]
    const firstTarget = batch.targetTokenIds[0]

    if (firstContext !== undefined && firstTarget !== undefined) {
        console.info('')
        console.info('Exemple décodé:')
        console.info(
            `  "${tokenizer.decode(firstContext)}" -> "${tokenizer.decode([firstTarget])}"`,
        )
    }
}

function inspectExample(exampleIndex: number): void {
    validateNonNegativeInteger(exampleIndex, 'index exemple')

    if (exampleIndex >= pipeline.trainExampleCount) {
        console.info(
            `Index trop grand. Dernier exemple train disponible: ${String(
                Math.max(0, pipeline.trainExampleCount - 1),
            )}.`,
        )

        return
    }

    const contextTokenIds = pipeline.trainTokenIds.slice(
        exampleIndex,
        exampleIndex + pipeline.contextLength,
    )
    const targetTokenId = pipeline.trainTokenIds[exampleIndex + pipeline.contextLength]

    if (targetTokenId === undefined) {
        console.info('Impossible de lire la cible pour cet exemple.')

        return
    }

    console.info('')
    console.info(`Exemple train ${String(exampleIndex)}:`)
    console.info(`  contexte tokens: [${contextTokenIds.join(', ')}]`)
    console.info(`  cible token: ${String(targetTokenId)}`)
    console.info(`  contexte décodé: "${tokenizer.decode(contextTokenIds)}"`)
    console.info(`  cible décodée: "${tokenizer.decode([targetTokenId])}"`)
    console.info('')
    console.info('Ce couple est une “bonne réponse” pour le module 18:')
    console.info('  le modèle recevra le contexte et devra prédire la cible.')
}

function inspectBatch(batchIndex: number): void {
    validateNonNegativeInteger(batchIndex, 'index batch')

    if (batchIndex >= pipeline.trainBatchCount) {
        console.info(
            `Index trop grand. Dernier batch train disponible: ${String(
                Math.max(0, pipeline.trainBatchCount - 1),
            )}.`,
        )

        return
    }

    const batch = getBatchByIndex(pipeline, batchIndex)

    if (batch === undefined) {
        console.info('Impossible de produire ce batch avec le contextLength actuel.')

        return
    }

    console.info('')
    console.info(`Batch train ${String(batchIndex)}:`)
    printBatchSummary(batch, pipeline)
    console.info(`  premier exemple global: ${String(batch.startExampleIndex)}`)
    console.info('')
    console.info('Exemples décodés du batch:')

    for (let rowIndex = 0; rowIndex < batch.inputTokenIds.length; rowIndex++) {
        const contextTokenIds = batch.inputTokenIds[rowIndex]
        const targetTokenId = batch.targetTokenIds[rowIndex]

        if (contextTokenIds !== undefined && targetTokenId !== undefined) {
            console.info(
                `  [${String(rowIndex)}] "${tokenizer.decode(contextTokenIds)}" -> "${tokenizer.decode(
                    [targetTokenId],
                )}"`,
            )
        }
    }
}

function updateContextLength(nextContextLength: number): void {
    validatePositiveInteger(nextContextLength, 'contextLength')

    currentContextLength = nextContextLength
    pipeline = createPipeline(currentContextLength)

    console.info('')
    console.info(`Context length mis à jour: ${String(currentContextLength)}`)
    console.info('Effet direct:')
    console.info('  - les contextes affichés sont plus courts ou plus longs;')
    console.info('  - le nombre d’exemples possibles change;')
    console.info('  - la mémoire d’un batch change aussi.')
    console.info('')
    printPipelineStats(pipeline)
}

function getBatchByIndex(
    nextPipeline: LongCorpusPipeline,
    batchIndex: number,
): NextTokenBatch | undefined {
    const startExampleIndex = batchIndex * nextPipeline.batchSize
    const iterator = iterateNextTokenBatches(nextPipeline.trainTokenIds.slice(startExampleIndex), {
        batchSize: nextPipeline.batchSize,
        contextLength: nextPipeline.contextLength,
    })
    const result = iterator.next()

    if (result.done === true) {
        return undefined
    }

    return {
        ...result.value,
        batchIndex,
        startExampleIndex,
    }
}

function estimateBatchBytes(batch: NextTokenBatch, nextContextLength: number): number {
    const int32Bytes = 4
    const inputValues = batch.inputTokenIds.length * nextContextLength
    const targetValues = batch.targetTokenIds.length

    return (inputValues + targetValues) * int32Bytes
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Mode interactif:')
    console.info('  example <index>  affiche un couple contexte -> cible')
    console.info('  batch <index>    affiche un batch décodé')
    console.info('  context <taille> recalcule la pipeline avec un autre contextLength')
    console.info('Appuie sur ENTRÉE pour exécuter une commande, ou sur ESC pour quitter.')
    console.info('')
    process.stdout.write('> ')

    let currentInput = ''

    process.stdin.setRawMode(true)
    process.stdin.resume()
    process.stdin.setEncoding('utf8')

    await new Promise<void>((resolve) => {
        const handleInput = (input: string): void => {
            if (input === '\u001B' || input === '\u0003') {
                process.stdin.off('data', handleInput)
                process.stdin.setRawMode(false)
                process.stdin.pause()
                console.info('')
                console.info('Démo terminée.')
                resolve()

                return
            }

            if (input === '\r' || input === '\n') {
                console.info('')
                handleCommand(currentInput)
                currentInput = ''
                console.info('')
                process.stdout.write('> ')

                return
            }

            if (input === '\u007F' || input === '\b') {
                currentInput = currentInput.slice(0, -1)

                return
            }

            currentInput += input
            process.stdout.write(input)
        }

        process.stdin.on('data', handleInput)
    })
}

function handleCommand(command: string): void {
    const trimmedCommand = command.trim()

    if (trimmedCommand.length === 0) {
        console.info('Commande vide.')

        return
    }

    const [commandName, rawValue] = trimmedCommand.split(/\s+/u)

    if (commandName === undefined) {
        console.info('Commande vide.')

        return
    }

    if (rawValue === undefined) {
        console.info('Ajoute une valeur. Exemple: example 12')

        return
    }

    const numericValue = Number(rawValue)

    if (!Number.isInteger(numericValue)) {
        console.info(`"${rawValue}" n’est pas un entier valide.`)

        return
    }

    try {
        if (commandName === 'example') {
            inspectExample(numericValue)

            return
        }

        if (commandName === 'batch') {
            inspectBatch(numericValue)

            return
        }

        if (commandName === 'context') {
            updateContextLength(numericValue)

            return
        }

        console.info(`Commande inconnue: ${commandName}`)
        console.info('Commandes disponibles: example, batch, context.')
    } catch (error) {
        console.info(formatError(error))
    }
}

function formatError(error: unknown): string {
    if (error instanceof Error) {
        return error.message
    }

    return 'Erreur inconnue.'
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`${name} doit être un entier strictement positif.`)
    }
}

function validateNonNegativeInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value < 0) {
        throw new Error(`${name} doit être un entier positif ou nul.`)
    }
}
