import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createNextTokenExamples } from '../08-training-loop-cpu/index.js'
import {
    createMinimalLanguageModel,
    trainMinimalLanguageModel,
} from '../09-minimal-trainable-language-model/index.js'
import { generateText, type TextGenerationResult } from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const contextLength = 4
const maxNewTokens = 40
const epochs = 30
const learningRate = 0.3
const defaultPrompt = 'bonj'

const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength })
const model = createMinimalLanguageModel({
    contextLength,
    vocabularySize: tokenizer.vocabularySize,
})

trainMinimalLanguageModel(model, examples, { epochs, learningRate })

console.info('Module 10 - Text generation greedy CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Créer le tokenizer')
console.info('3. Créer le dataset de tokens')
console.info('4. Créer des exemples contexte -> cible')
console.info('5. Entraîner le modèle minimal du module 9')
console.info('6. Générer plusieurs tokens avec greedy decoding')
console.info('')
console.info('Fil conducteur de la génération:')
console.info('  prompt -> ids -> contexte -> modèle -> meilleur token -> append -> boucle')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du corpus:')
console.info(rawText)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Tokens train: ${String(dataset.trainTokenCount)}`)
console.info(`Longueur de contexte: ${String(contextLength)}`)
console.info(`Tokens générés: ${String(maxNewTokens)}`)
console.info(`Epochs d'entraînement: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('')
console.info('Idée clé:')
console.info('  Le module 9 prédit un seul prochain token.')
console.info('  Le module 10 réutilise cette prédiction en boucle pour construire une séquence.')
console.info('')
console.info('Limite importante:')
console.info('  Greedy decoding choisit toujours le token le plus probable.')
console.info('  C’est déterministe et lisible, mais cela peut vite devenir répétitif.')
console.info('  Il n’y a pas encore de token <eos>: la génération s’arrête avec maxNewTokens.')
console.info('')

showGeneration(defaultPrompt)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        'Mode non interactif détecté: lance cette démo dans un terminal pour tester tes propres prompts.',
    )
}

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

function showGeneration(prompt: string): void {
    validatePromptLength(prompt)

    const result = generateText(model, tokenizer, prompt, { maxNewTokens })

    console.info(`Prompt initial: "${prompt}"`)
    console.info(`Texte final: "${result.text}"`)
    console.info('')
    console.info('Étapes de génération:')
    printGenerationSteps(result)
}

function printGenerationSteps(result: TextGenerationResult): void {
    for (const generationStep of result.steps) {
        console.info(
            `  étape ${String(generationStep.step).padStart(2, ' ')} | contexte "${decodeTokenIds(
                generationStep.contextTokenIds,
            )}" -> "${formatToken(generationStep.predictedTokenId)}" | texte "${decodeTokenIds(
                generationStep.tokenIdsAfterPrediction,
            )}"`,
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

                if (currentInput.length === 0) {
                    console.info('Aucun prompt saisi.')
                } else {
                    try {
                        showGeneration(currentInput)
                    } catch (error) {
                        console.info(toEducationalErrorMessage(currentInput, error))
                    }
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
