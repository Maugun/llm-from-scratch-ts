import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createNextTokenExamples } from '../08-training-loop-cpu/index.js'
import {
    computeMiniTransformerAverageLoss,
    createTrainableMiniTransformer,
    disposeTrainableMiniTransformer,
    generateMiniTransformerText,
    predictMiniTransformerProbabilities,
    trainMiniTransformer,
} from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const contextLength = 4
const embeddingDimension = 8
const feedForwardDimension = 16
const epochs = 40
const learningRate = 0.03
const maxNewTokens = 40
const previewStepCount = 10
const topPredictionCount = 5
const defaultPrompt = 'bonj'

const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength })
const model = createTrainableMiniTransformer({
    contextLength,
    embeddingDimension,
    feedForwardDimension,
    seed: 14,
    vocabularySize: tokenizer.vocabularySize,
})

console.info('Module 14 - Mini Transformer entraînable + génération greedy')
console.info('')
console.info('But du module:')
console.info(
    '  Réunir embeddings, positions, self-attention, feed-forward, autograd et génération.',
)
console.info('')
console.info('Différence avec le module 13:')
console.info('  Module 13: le contexte est aplati puis projeté directement.')
console.info('  Module 14: les positions communiquent avec une self-attention causale entraînable.')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le corpus')
console.info('2. Créer le tokenizer')
console.info('3. Créer les exemples contexte -> cible')
console.info('4. Embeddings token + position')
console.info('5. Self-attention causale')
console.info('6. Résiduel puis feed-forward')
console.info('7. Prédiction du prochain token')
console.info('8. Génération greedy en boucle')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du corpus:')
console.info(rawText)
console.info('')
console.info('Shapes principales:')
console.info(`  contexte tokenisé: [${String(contextLength)}]`)
console.info(
    `  tokenEmbeddings: [${String(tokenizer.vocabularySize)}, ${String(embeddingDimension)}]`,
)
console.info(`  positionEmbeddings: [${String(contextLength)}, ${String(embeddingDimension)}]`)
console.info(`  scores attention: [${String(contextLength)}, ${String(contextLength)}]`)
console.info(
    `  feed-forward: ${String(embeddingDimension)} -> ${String(feedForwardDimension)} -> ${String(embeddingDimension)}`,
)
console.info(`  logits: [${String(tokenizer.vocabularySize)}]`)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Exemples d'entraînement: ${String(examples.length)}`)
console.info(`Epochs: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('Optimizer: Adam')
console.info(`Tokens générés: ${String(maxNewTokens)}`)
console.info('')
console.info('Quelques exemples contexte -> cible:')

for (const example of examples.slice(0, 5)) {
    console.info(
        `  "${decodeTokenIds(example.inputTokenIds)}" -> "${formatToken(example.targetTokenId)}"`,
    )
}

const initialLoss = computeMiniTransformerAverageLoss(model, examples)

console.info('')
console.info('Avant entraînement:')
console.info(`  loss: ${initialLoss.toFixed(4)}`)
console.info(`  perplexité: ${Math.exp(initialLoss).toFixed(2)}`)
printTopPredictions(defaultPrompt)

const history = trainMiniTransformer(model, examples, {
    epochs,
    learningRate,
})

console.info('')
console.info('Pendant l’entraînement:')

for (const metrics of history.epochs) {
    if (metrics.epoch === 1 || metrics.epoch % 5 === 0 || metrics.epoch === epochs) {
        console.info(
            `  epoch ${String(metrics.epoch).padStart(2, ' ')} | loss ${metrics.averageLoss.toFixed(
                4,
            )} | perplexité ${metrics.perplexity.toFixed(2)}`,
        )
    }
}

console.info('')
console.info('Après entraînement:')
console.info(`  loss finale: ${history.finalLoss.toFixed(4)}`)
console.info(`  perplexité finale: ${Math.exp(history.finalLoss).toFixed(2)}`)
printTopPredictions(defaultPrompt)
console.info('')
printGeneration(defaultPrompt)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        'Mode non interactif détecté: lance cette démo dans un terminal pour tester tes propres prompts.',
    )
}

disposeTrainableMiniTransformer(model)

function decodeTokenIds(tokenIds: readonly number[]): string {
    return tokenizer.decode([...tokenIds])
}

function formatToken(tokenId: number): string {
    const text = decodeTokenIds([tokenId])

    if (text === '\n') {
        return '\\n'
    }

    return text
}

function printTopPredictions(contextText: string): void {
    const inputTokenIds = tokenizer.encode(contextText)
    const probabilities = predictMiniTransformerProbabilities(model, inputTokenIds)
    const topPredictions = probabilities
        .map((probability, tokenId) => ({ probability, tokenId }))
        .sort((left, right) => right.probability - left.probability)
        .slice(0, topPredictionCount)

    console.info(`  contexte "${contextText}"`)

    for (const prediction of topPredictions) {
        console.info(
            `    "${formatToken(prediction.tokenId)}" -> ${(prediction.probability * 100).toFixed(
                2,
            )}%`,
        )
    }
}

function printGeneration(prompt: string): void {
    const result = generateMiniTransformerText(model, tokenizer, prompt, { maxNewTokens })

    console.info(`Génération greedy depuis "${prompt}":`)
    console.info(`  texte final: "${result.text}"`)
    console.info('  premières étapes:')

    for (const step of result.steps.slice(0, previewStepCount)) {
        console.info(
            `    étape ${String(step.step).padStart(2, ' ')} | contexte "${decodeTokenIds(
                step.contextTokenIds,
            )}" -> "${formatToken(step.selectedTokenId)}"`,
        )
    }
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info(`Saisis un prompt d'au moins ${String(contextLength)} caractères du corpus.`)
    console.info('Appuie sur ENTRÉE pour générer, ou sur ESC pour quitter.')
    console.info('')

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

                try {
                    validatePromptLength(currentInput)
                    printGeneration(currentInput)
                } catch (error) {
                    console.info(toEducationalErrorMessage(currentInput, error))
                }

                currentInput = ''
                console.info('')
                console.info('Saisis un autre prompt, ou appuie sur ESC pour quitter.')

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

function toEducationalErrorMessage(prompt: string, error: unknown): string {
    if (error instanceof Error) {
        return `Impossible de générer après "${prompt}": ${error.message}`
    }

    return `Impossible de générer après "${prompt}".`
}

function validatePromptLength(prompt: string): void {
    const promptLength = Array.from(prompt).length

    if (promptLength < contextLength) {
        throw new Error(
            `Le prompt doit contenir au moins ${String(
                contextLength,
            )} caractères. Nombre reçu: ${String(promptLength)}.`,
        )
    }
}
