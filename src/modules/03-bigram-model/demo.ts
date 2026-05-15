import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createBigramModel } from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const model = createBigramModel(dataset.trainTokenIds, tokenizer.vocabularySize)
const defaultCharacter = 'l'

console.info('Module 3 - Bigram model CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Creer le tokenizer')
console.info('3. Creer le dataset de tokens')
console.info('4. Construire le modele bigramme avec la partie entrainement du dataset')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du fichier:')
console.info(rawText)
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caracteres`)
console.info(`Tokens utilises pour apprendre les transitions: ${String(dataset.trainTokenCount)}`)
console.info(`Transitions observees: ${String(model.totalTransitions)}`)
console.info('')

showPredictionForCharacter(defaultCharacter)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        "Mode non interactif detecte: lance cette demo dans un terminal pour choisir d'autres lettres.",
    )
}

function encodeSingleToken(character: string): number {
    const tokenId = tokenizer.encode(character)[0]

    if (tokenId === undefined) {
        throw new Error(`Impossible d'encoder le caractere "${character}" dans la demo.`)
    }

    return tokenId
}

function showPredictionForCharacter(currentCharacter: string): void {
    const currentTokenId = encodeSingleToken(currentCharacter)
    const probabilities = model.getNextTokenProbabilities(currentTokenId)
    const predictedTokenId = model.predictMostLikelyNextToken(currentTokenId)

    console.info(`Token courant: "${currentCharacter}" -> id ${String(currentTokenId)}`)
    console.info('Probabilites non nulles pour le prochain token:')

    for (const [tokenId, probability] of probabilities.entries()) {
        if (probability > 0) {
            console.info(
                `  "${tokenizer.decode([tokenId])}" -> ${probability.toFixed(3)} (${String(
                    model.getTransitionCount(currentTokenId, tokenId),
                )} occurrence(s))`,
            )
        }
    }

    console.info('')

    if (predictedTokenId === undefined) {
        console.info('Aucune prediction possible pour ce token.')
    } else {
        const predictedCharacter = tokenizer.decode([predictedTokenId])

        console.info(
            `Prediction la plus probable apres "${currentCharacter}": "${predictedCharacter}"`,
        )
        console.info(
            `Le modele ne regarde que le dernier token: si le texte finit par "${currentCharacter}", il choisirait ensuite "${predictedCharacter}".`,
        )
    }
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Choisis une lettre du vocabulaire pour inspecter ses transitions.')
    console.info('Appuie sur ESC pour quitter.')
    console.info('')

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
                console.info('Demo terminee.')
                resolve()

                return
            }

            if (input === '\r' || input === '\n') {
                return
            }

            console.info('')

            try {
                showPredictionForCharacter(input)
            } catch {
                console.info(
                    `Le caractere "${input}" n'est pas dans le vocabulaire du corpus. Essaie une autre lettre.`,
                )
            }

            console.info('')
            console.info('Choisis une autre lettre, ou appuie sur ESC pour quitter.')
        }

        process.stdin.on('data', handleInput)
    })
}
