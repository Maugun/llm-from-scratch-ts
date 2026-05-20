import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createNextTokenExamples } from '../08-training-loop-cpu/index.js'
import {
    createMinimalLanguageModel,
    trainMinimalLanguageModel,
} from '../09-minimal-trainable-language-model/index.js'
import {
    generateTextWithSampling,
    type SamplingGenerationOptions,
    type SamplingTextGenerationResult,
} from './index.js'

type DemoStrategy = {
    readonly label: string
    readonly intuition: string
    readonly options: SamplingGenerationOptions
}

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const contextLength = 4
const maxNewTokens = 40
const previewStepCount = 8
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

const demoStrategies: readonly DemoStrategy[] = [
    {
        intuition: 'Choisit toujours le token le plus probable. Stable, mais souvent répétitif.',
        label: 'Greedy',
        options: {
            maxNewTokens,
            seed: 1,
            strategy: 'greedy',
        },
    },
    {
        intuition: 'Température basse: la distribution devient plus pointue, donc plus prudente.',
        label: 'Temperature 0.7',
        options: {
            maxNewTokens,
            seed: 2,
            strategy: 'temperature',
            temperature: 0.7,
        },
    },
    {
        intuition: 'Température haute: la distribution devient plus plate, donc plus variable.',
        label: 'Temperature 1.4',
        options: {
            maxNewTokens,
            seed: 3,
            strategy: 'temperature',
            temperature: 1.4,
        },
    },
    {
        intuition: 'Top-k: on garde les 3 candidats les plus probables, puis on tire parmi eux.',
        label: 'Top-k 3',
        options: {
            maxNewTokens,
            seed: 4,
            strategy: 'topK',
            temperature: 1,
            topK: 3,
        },
    },
]

console.info('Module 11 - Sampling strategies CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Créer le tokenizer')
console.info('3. Entraîner le modèle minimal du module 9')
console.info('4. Générer du texte avec plusieurs stratégies de sélection')
console.info('')
console.info('Différence clé avec le module 10:')
console.info('  Module 10: greedy decoding, toujours le token le plus probable.')
console.info('  Module 11: sampling, on peut tirer un token selon une distribution transformée.')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du corpus:')
console.info(rawText)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Longueur de contexte: ${String(contextLength)}`)
console.info(`Tokens générés par stratégie: ${String(maxNewTokens)}`)
console.info(`Epochs d'entraînement: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('')
console.info('Idées clés:')
console.info('  Greedy = déterministe.')
console.info('  Sampling = probabiliste.')
console.info('  Seed = tirage reproductible pour garder les démos et tests stables.')
console.info('')

showStrategyComparison(defaultPrompt)

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

function showStrategyComparison(prompt: string): void {
    validatePromptLength(prompt)

    console.info(`Prompt: "${prompt}"`)
    console.info('')

    for (const strategy of demoStrategies) {
        const result = generateTextWithSampling(model, tokenizer, prompt, strategy.options)

        console.info(strategy.label)
        console.info(`  Intuition: ${strategy.intuition}`)
        console.info(`  Texte final: "${result.text}"`)
        console.info('  Premières étapes:')
        printStepPreview(result)
        console.info('')
    }
}

function printStepPreview(result: SamplingTextGenerationResult): void {
    for (const generationStep of result.steps.slice(0, previewStepCount)) {
        console.info(
            `    étape ${String(generationStep.step).padStart(2, ' ')} | contexte "${decodeTokenIds(
                generationStep.contextTokenIds,
            )}" -> "${formatToken(generationStep.selectedTokenId)}" | stratégie ${
                generationStep.strategy
            }`,
        )
    }
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info(`Saisis un prompt d'au moins ${String(contextLength)} caractères du corpus.`)
    console.info('Appuie sur ENTRÉE pour comparer les stratégies, ou sur ESC pour quitter.')
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
                        showStrategyComparison(currentInput)
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
