import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import {
    createNextTokenExamples,
    createTrainableTokenBiasModel,
    crossEntropyLoss,
    perplexityFromLoss,
    trainNextTokenModel,
    type NextTokenExample,
    type TrainableNextTokenModel,
} from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const contextLength = 4
const epochs = 20
const learningRate = 0.5
const topTokenCount = 5

const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const examples = createNextTokenExamples(dataset.trainTokenIds, { contextLength })
const model = createTrainableTokenBiasModel({ vocabularySize: tokenizer.vocabularySize })
const initialProbabilities = model.predictNextTokenProbabilities()
const initialLoss = averageLoss(model, examples)
const history = trainNextTokenModel(model, examples, { epochs, learningRate })
const initialPerplexity = perplexityFromLoss(initialLoss)
const lossImprovement = initialLoss - history.finalLoss
const lossImprovementRatio = lossImprovement / initialLoss

console.info('Module 8 - Training loop CPU pédagogique')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Créer le tokenizer')
console.info('3. Créer le dataset de tokens')
console.info('4. Créer des exemples contexte -> cible')
console.info('5. Prédire avec des logits')
console.info('6. Calculer la loss')
console.info('7. Corriger les logits avec une descente de gradient')
console.info('')
console.info('Fil conducteur de la démo:')
console.info('  données -> exemples corrigés -> prédiction -> loss -> correction -> métriques')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du corpus:')
console.info(rawText)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Tokens train: ${String(dataset.trainTokenCount)}`)
console.info(`Exemples créés: ${String(examples.length)}`)
console.info(`Longueur de contexte: ${String(contextLength)}`)
console.info(`Epochs: ${String(epochs)}`)
console.info(`Learning rate: ${String(learningRate)}`)
console.info('')
console.info('Définitions utiles:')
console.info('  Logits: scores bruts du modèle, avant conversion en probabilités.')
console.info('  Softmax: transforme les logits en probabilités dont la somme vaut 1.')
console.info("  Loss: score d'erreur que l'entraînement cherche à faire baisser.")
console.info('  Perplexité: exp(loss), une autre lecture de l’incertitude moyenne du modèle.')
console.info('')
console.info('Limite importante:')
console.info(
    '  Ce mini modèle ignore encore le contexte. Il apprend seulement quels tokens sont probables globalement.',
)
console.info(
    "  La démo reste linéaire pour éviter de faire croire qu'un contexte saisi influencerait la prédiction.",
)
console.info('')

console.info('Étape 1 - Construire des exemples corrigés')
console.info(
    'On découpe les tokens du corpus en exercices: avec ce contexte, le token attendu est celui-ci.',
)
printExamples(examples.slice(0, 5))
console.info('Étape 2 - Observer le modèle avant entraînement')
console.info(
    'Au départ, tous les logits valent 0: le modèle ne préfère aucun token, donc les probabilités sont uniformes.',
)
printTopTokens('Top tokens avant entraînement:', initialProbabilities)
console.info(`Loss avant entraînement: ${initialLoss.toFixed(4)}`)
console.info(`Perplexité avant entraînement: ${initialPerplexity.toFixed(2)}`)
console.info(
    'La loss mesure si le modèle donne assez de probabilité aux bonnes réponses attendues.',
)
console.info('')
console.info('Étape 3 - Entraîner')
console.info(
    'À chaque epoch, le modèle repasse sur les exemples et ajuste ses logits pour réduire la loss.',
)
printTrainingHistory()
console.info('')
console.info('Étape 4 - Observer le modèle après entraînement')
console.info(`Loss finale recalculée: ${history.finalLoss.toFixed(4)}`)
console.info(`Perplexité finale: ${perplexityFromLoss(history.finalLoss).toFixed(2)}`)
console.info(
    `Amélioration de loss: -${lossImprovement.toFixed(4)} (${(lossImprovementRatio * 100).toFixed(
        2,
    )} %)`,
)
console.info(
    'Cette baisse reste modeste car le modèle ne peut apprendre que la fréquence globale des tokens.',
)
console.info('')
console.info(
    'Les tokens fréquents dans les cibles du corpus ont maintenant des logits plus élevés, donc des probabilités plus fortes.',
)
printTopTokens('Top tokens après entraînement:', model.predictNextTokenProbabilities())
console.info(
    "Conclusion: le modèle a appris une distribution globale des tokens, mais il n'utilise pas encore le contexte.",
)
console.info(
    "Le prochain vrai saut sera d'apprendre une distribution conditionnée par le contexte.",
)

function averageLoss(
    trainableModel: TrainableNextTokenModel,
    nextTokenExamples: readonly NextTokenExample[],
): number {
    const probabilities = trainableModel.predictNextTokenProbabilities()
    const lossSum = nextTokenExamples.reduce(
        (sum, example) => sum + crossEntropyLoss(probabilities, example.targetTokenId),
        0,
    )

    return lossSum / nextTokenExamples.length
}

function decodeTokenIds(tokenIds: readonly number[]): string {
    return tokenizer.decode([...tokenIds])
}

function printExamples(nextTokenExamples: readonly NextTokenExample[]): void {
    console.info('Quelques exemples contexte -> cible:')

    for (const example of nextTokenExamples) {
        console.info(
            `  "${decodeTokenIds(example.inputTokenIds)}" -> "${decodeTokenIds([
                example.targetTokenId,
            ])}"`,
        )
    }

    console.info('')
}

function printTopTokens(title: string, probabilities: readonly number[]): void {
    console.info(title)

    const rankedTokens = probabilities
        .map((probability, tokenId) => ({ probability, tokenId }))
        .sort((left, right) => right.probability - left.probability)
        .slice(0, topTokenCount)

    for (const { probability, tokenId } of rankedTokens) {
        console.info(
            `  "${decodeTokenIds([tokenId])}" token ${String(tokenId)} -> ${(
                probability * 100
            ).toFixed(2)} %`,
        )
    }

    console.info('')
}

function printTrainingHistory(): void {
    console.info('Loss par epoch:')

    for (const metrics of history.epochs) {
        console.info(
            `  epoch ${String(metrics.epoch).padStart(2, ' ')} | loss ${metrics.averageLoss.toFixed(
                4,
            )} | perplexité ${metrics.perplexity.toFixed(2)}`,
        )
    }
}
