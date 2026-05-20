import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import {
    createNextTokenExamples,
    createTrainableTokenBiasModel,
    perplexityFromLoss,
    trainNextTokenModel,
    type NextTokenExample,
} from '../08-training-loop-cpu/index.js'
import {
    computeAverageLoss,
    createMinimalLanguageModel,
    predictMostLikelyNextToken,
    predictNextTokenProbabilities,
    trainMinimalLanguageModel,
} from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const contextLength = 4
const epochs = 30
const learningRate = 0.3
const topTokenCount = 5
const fixedContextToInspect = 'bonj'

const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength })
const trainingExamples = examples.filter(
    (example) => example.inputTokenIds.length === contextLength,
)
const globalModel = createTrainableTokenBiasModel({ vocabularySize: tokenizer.vocabularySize })
const model = createMinimalLanguageModel({
    contextLength,
    vocabularySize: tokenizer.vocabularySize,
})

const initialLoss = computeAverageLoss(model, trainingExamples)
const globalHistory = trainNextTokenModel(globalModel, trainingExamples, { epochs, learningRate })
const history = trainMinimalLanguageModel(model, trainingExamples, { epochs, learningRate })
const globalImprovementRatio = (initialLoss - globalHistory.finalLoss) / initialLoss
const conditionalImprovementRatio = (initialLoss - history.finalLoss) / initialLoss

console.info('Module 9 - Modèle de langage minimal entraînable CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Créer le tokenizer')
console.info('3. Créer le dataset de tokens')
console.info('4. Créer des exemples contexte -> cible')
console.info('5. Calculer des logits conditionnés par le contexte')
console.info('6. Convertir les logits en probabilités avec softmax')
console.info('7. Calculer la loss et corriger les poids du contexte')
console.info('')
console.info('Fil conducteur de la démo:')
console.info(
    '  contexte -> logits conditionnés -> probabilités -> loss -> correction -> prédiction',
)
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du corpus:')
console.info(rawText)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Tokens train: ${String(dataset.trainTokenCount)}`)
console.info(`Exemples créés: ${String(trainingExamples.length)}`)
console.info(`Longueur de contexte: ${String(contextLength)}`)
console.info(`Epochs: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('')
console.info('Différence clé avec le module 8:')
console.info('  Module 8: P(nextToken), une distribution globale.')
console.info(
    '  Module 9: P(nextToken | contexte), une distribution qui dépend des tokens en entrée.',
)
console.info('')
console.info('Forme du modèle:')
console.info('  outputBias[next] apprend la préférence globale pour un prochain token.')
console.info('  contextWeights[position][tokenContexte][next] apprend une influence contextuelle.')
console.info('  logit(next) = outputBias(next) + moyenne des contributions du contexte.')
console.info('')

console.info('Étape 1 - Construire des exemples corrigés')
console.info(
    'On fabrique les mêmes exemples que dans le module 8, mais cette fois le contexte va vraiment servir au modèle.',
)
printExamples(trainingExamples.slice(0, 5))
console.info('Étape 2 - Observer le modèle avant entraînement')
console.info(
    'Tous les poids commencent à 0: avant apprentissage, le modèle conditionné prédit comme une distribution uniforme.',
)
console.info('Avant entraînement:')
console.info(`  loss: ${initialLoss.toFixed(4)}`)
console.info(`  perplexité: ${perplexityFromLoss(initialLoss).toFixed(2)}`)
console.info('')
console.info('Étape 3 - Entraîner le modèle conditionné')
console.info(
    'À chaque exemple, seuls les poids liés aux tokens présents dans le contexte observé sont corrigés.',
)
console.info('Entraînement du modèle conditionné:')
printTrainingHistory(trainingExamples.length)
console.info('')
console.info('Étape 4 - Comparer modèle global et modèle conditionné')
console.info('Après entraînement:')
console.info(`  loss globale module 8: ${globalHistory.finalLoss.toFixed(4)}`)
console.info(`  amélioration globale: ${(globalImprovementRatio * 100).toFixed(2)} %`)
console.info(`  loss conditionnée module 9: ${history.finalLoss.toFixed(4)}`)
console.info(`  amélioration conditionnée: ${(conditionalImprovementRatio * 100).toFixed(2)} %`)
console.info(`  perplexité conditionnée: ${perplexityFromLoss(history.finalLoss).toFixed(2)}`)
console.info(
    '  Le modèle 9 peut descendre plus bas parce qu’il utilise le contexte au lieu de seulement compter les tokens fréquents.',
)
console.info('')
console.info('Étape 5 - Inspecter des prédictions contextuelles')
printContextPredictions()
console.info('Conclusion:')
console.info('  Le modèle utilise maintenant le contexte pour changer sa distribution de sortie.')
console.info('  Il ne génère pas encore du texte complet: il prédit seulement le prochain token.')

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        'Mode non interactif détecté: lance cette démo dans un terminal pour tester tes propres contextes.',
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

function printExamples(nextTokenExamples: readonly NextTokenExample[]): void {
    console.info('Quelques exemples contexte -> cible:')

    for (const example of nextTokenExamples) {
        console.info(
            `  "${decodeTokenIds(example.inputTokenIds)}" -> "${formatToken(
                example.targetTokenId,
            )}"`,
        )
    }

    console.info('')
}

function printTrainingHistory(exampleCount: number): void {
    console.info(`Le modèle repasse sur ${String(exampleCount)} exemples à chaque epoch.`)

    for (const metrics of history.epochs) {
        console.info(
            `  epoch ${String(metrics.epoch).padStart(2, ' ')} | loss ${metrics.averageLoss.toFixed(
                4,
            )} | perplexité ${metrics.perplexity.toFixed(2)}`,
        )
    }
}

function printContextPredictions(): void {
    console.info('Prédiction pour un contexte fixe:')

    printPredictionForContext(fixedContextToInspect, '  ')
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info(
        `Saisis un contexte de ${String(contextLength)} caractères présents dans le corpus.`,
    )
    console.info('Appuie sur ENTRÉE pour prédire le prochain token, ou sur ESC pour quitter.')
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
                    console.info('Aucun contexte saisi.')
                } else {
                    try {
                        printPredictionForContext(currentInput, '')
                    } catch (error) {
                        console.info(toEducationalErrorMessage(currentInput, error))
                    }
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

function printPredictionForContext(context: string, indentation: string): void {
    const inputTokenIds = tokenizer.encode(context)

    if (Array.from(context).length !== contextLength) {
        throw new Error(`Le contexte doit contenir exactement ${String(contextLength)} caractères.`)
    }

    const globalProbabilities = globalModel.predictNextTokenProbabilities()
    const conditionalProbabilities = predictNextTokenProbabilities(model, inputTokenIds)
    const predictedTokenId = predictMostLikelyNextToken(model, inputTokenIds)

    console.info(`${indentation}contexte "${context}"`)
    console.info(`${indentation}  meilleur token conditionné: "${formatToken(predictedTokenId)}"`)
    console.info(`${indentation}  modèle global:`)
    printTopTokens(globalProbabilities, `${indentation}    `)
    console.info(`${indentation}  modèle conditionné:`)
    printTopTokens(conditionalProbabilities, `${indentation}    `)
}

function toEducationalErrorMessage(context: string, error: unknown): string {
    if (error instanceof Error) {
        return `Impossible de prédire après "${context}": ${error.message}`
    }

    return `Impossible de prédire après "${context}".`
}

function printTopTokens(probabilities: readonly number[], indentation: string): void {
    const rankedTokens = probabilities
        .map((probability, tokenId) => ({ probability, tokenId }))
        .sort((left, right) => right.probability - left.probability)
        .slice(0, topTokenCount)

    for (const { probability, tokenId } of rankedTokens) {
        console.info(
            `${indentation}"${formatToken(tokenId)}" token ${String(tokenId)} -> ${(
                probability * 100
            ).toFixed(2)} %`,
        )
    }
}
