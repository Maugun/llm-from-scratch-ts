import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createNextTokenExamples } from '../08-training-loop-cpu/index.js'
import {
    computeTfjsNextTokenAverageLoss,
    createTfjsNextTokenModel,
    disposeTfjsNextTokenModel,
    predictNextTokenProbabilities,
    trainTfjsNextTokenModel,
} from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const contextLength = 4
const embeddingDimension = 8
const epochs = 40
const learningRate = 0.1
const defaultContext = 'bonj'
const topPredictionCount = 5

const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength })
const model = createTfjsNextTokenModel({
    contextLength,
    embeddingDimension,
    seed: 13,
    vocabularySize: tokenizer.vocabularySize,
})

console.info('Module 13 - Modèle next-token TensorFlow.js')
console.info('')
console.info('But du module:')
console.info('  Appliquer TensorFlow.js à une vraie tâche de langage: prédire le prochain token.')
console.info('')
console.info('Différence avec le module 9:')
console.info('  Module 9: poids JavaScript et gradients écrits à la main.')
console.info('  Module 13: tf.Variable et gradients calculés automatiquement par TensorFlow.js.')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le corpus')
console.info('2. Créer le tokenizer')
console.info('3. Transformer le texte en exemples contexte -> cible')
console.info('4. Transformer les tokens en embeddings entraînables')
console.info('5. Projeter le contexte vers des logits de vocabulaire')
console.info('6. Calculer la cross-entropy')
console.info("7. Laisser l'autograd mettre à jour les variables")
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
console.info(`  contexte aplati: [${String(contextLength * embeddingDimension)}]`)
console.info(`  logits: [${String(tokenizer.vocabularySize)}]`)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Exemples d'entraînement: ${String(examples.length)}`)
console.info(`Epochs: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('Optimizer: Adam')
console.info('')
console.info('Quelques exemples contexte -> cible:')

for (const example of examples.slice(0, 5)) {
    console.info(
        `  "${decodeTokenIds(example.inputTokenIds)}" -> "${formatToken(example.targetTokenId)}"`,
    )
}

const initialLoss = computeTfjsNextTokenAverageLoss(model, examples)

console.info('')
console.info('Avant entraînement:')
console.info(`  loss: ${initialLoss.toFixed(4)}`)
console.info(`  perplexité: ${Math.exp(initialLoss).toFixed(2)}`)
printTopPredictions(defaultContext)

const history = trainTfjsNextTokenModel(model, examples, {
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
printTopPredictions(defaultContext)

console.info('')
console.info('Cap vers la fin du projet:')
console.info('  Ce module reste tiny pour comprendre les shapes et l’autograd.')
console.info('  Le dernier module visera une pipeline plus réaliste: corpus 5-20 MB,')
console.info('  contexte 128, 1M-10M paramètres et 2-4 layers.')

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        'Mode non interactif détecté: lance cette démo dans un terminal pour tester tes propres contextes.',
    )
}

disposeTfjsNextTokenModel(model)

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
    const probabilities = predictNextTokenProbabilities(model, inputTokenIds)
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

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info(`Saisis un contexte de ${String(contextLength)} caractères du corpus.`)
    console.info('Appuie sur ENTRÉE pour voir les prédictions, ou sur ESC pour quitter.')
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
                    validateContextText(currentInput)
                    printTopPredictions(currentInput)
                } catch (error) {
                    console.info(toEducationalErrorMessage(currentInput, error))
                }

                currentInput = ''
                console.info('')
                console.info('Saisis un autre contexte, ou appuie sur ESC pour quitter.')

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

function toEducationalErrorMessage(contextText: string, error: unknown): string {
    if (error instanceof Error) {
        return `Impossible de prédire après "${contextText}": ${error.message}`
    }

    return `Impossible de prédire après "${contextText}".`
}

function validateContextText(contextText: string): void {
    const contextTextLength = Array.from(contextText).length

    if (contextTextLength !== contextLength) {
        throw new Error(
            `Le contexte doit contenir exactement ${String(
                contextLength,
            )} caractères. Nombre reçu: ${String(contextTextLength)}.`,
        )
    }
}
