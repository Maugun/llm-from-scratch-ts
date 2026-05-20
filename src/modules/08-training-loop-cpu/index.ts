export type NextTokenExample = {
    /**
     * Contexte fourni au modèle.
     *
     * Dans un vrai LLM, le modèle utiliserait ces ids pour prédire la cible. Dans ce module,
     * on les conserve pour comprendre la forme des données, même si le mini modèle ignore
     * volontairement le contexte.
     */
    readonly inputTokenIds: readonly number[]

    /**
     * Token que le modèle doit apprendre à prédire.
     */
    readonly targetTokenId: number
}

export type NextTokenExampleOptions = {
    /**
     * Nombre de tokens donnés comme contexte avant la cible.
     */
    readonly contextLength: number
}

export type TrainableTokenBiasModelOptions = {
    readonly vocabularySize: number
}

export type TrainableNextTokenModel = {
    readonly vocabularySize: number

    /**
     * Paramètres entraînables du mini modèle.
     *
     * Un logit est un score brut avant softmax. Ici, on stocke un score par token du vocabulaire.
     * Plus le logit d'un token monte, plus le softmax donnera une probabilité élevée à ce token.
     */
    readonly logits: number[]

    readonly predictNextTokenProbabilities: () => readonly number[]
}

export type TrainingOptions = {
    readonly epochs: number
    readonly learningRate: number
}

export type TrainingEpochMetrics = {
    readonly epoch: number
    readonly averageLoss: number
    readonly perplexity: number
}

export type TrainingHistory = {
    readonly initialLoss: number
    readonly finalLoss: number
    readonly epochs: readonly TrainingEpochMetrics[]
}

/**
 * Transforme une longue séquence de tokens en exemples de prédiction du prochain token.
 *
 * Le tableau retourné devient un mini dataset d'entraînement:
 * - inputTokenIds contient ce que le modèle reçoit;
 * - targetTokenId contient la prédiction attendue, donc la "bonne réponse" utilisée pour
 *   calculer la loss et corriger les paramètres.
 *
 * Exemple avec contextLength = 3:
 * tokens:  [10, 11, 12, 13]
 * entrée:  [10, 11, 12]
 * cible:   13
 */
export function createNextTokenExamples(
    tokenIds: readonly number[],
    options: NextTokenExampleOptions,
): readonly NextTokenExample[] {
    validatePositiveInteger(options.contextLength, 'contextLength')

    const examples: NextTokenExample[] = []

    for (let targetIndex = options.contextLength; targetIndex < tokenIds.length; targetIndex++) {
        const targetTokenId = readTokenIdAt(tokenIds, targetIndex)
        const inputTokenIds = tokenIds.slice(targetIndex - options.contextLength, targetIndex)

        examples.push({
            inputTokenIds,
            targetTokenId,
        })
    }

    return examples
}

/**
 * Convertit des logits en probabilités.
 *
 * Les logits sont des scores libres: ils peuvent être négatifs, positifs, grands ou petits.
 * Le softmax les transforme en distribution:
 * - chaque valeur devient positive;
 * - la somme vaut 1;
 * - le plus grand logit reçoit la plus grande probabilité.
 */
export function softmax(logits: readonly number[]): readonly number[] {
    if (logits.length === 0) {
        throw new Error('softmax attend au moins un logit.')
    }

    validateFiniteValues(logits, 'logits')

    // Soustraire le maximum ne change pas le résultat mathématique du softmax, mais évite
    // d'appeler Math.exp sur des valeurs trop grandes. C'est une petite astuce de stabilité.
    const maxLogit = Math.max(...logits)
    const exponentials = logits.map((logit) => Math.exp(logit - maxLogit))
    const exponentialSum = exponentials.reduce((sum, value) => sum + value, 0)

    return exponentials.map((value) => value / exponentialSum)
}

/**
 * Mesure l'erreur de prédiction pour une cible.
 *
 * Intuition:
 * - si le bon token a une probabilité proche de 1, la loss est proche de 0;
 * - si le bon token a une probabilité faible, la loss devient grande.
 */
export function crossEntropyLoss(probabilities: readonly number[], targetTokenId: number): number {
    validateProbabilityDistributionShape(probabilities)
    validateTokenId(targetTokenId, probabilities.length, 'targetTokenId')

    const targetProbability = readNumberAt(probabilities, targetTokenId)

    return -Math.log(Math.max(targetProbability, Number.MIN_VALUE))
}

export function perplexityFromLoss(loss: number): number {
    if (!Number.isFinite(loss) || loss < 0) {
        throw new Error(
            `loss doit être un nombre fini positif ou nul. Valeur reçue: ${String(loss)}.`,
        )
    }

    return Math.exp(loss)
}

/**
 * Crée un mini modèle entraînable.
 *
 * Ce modèle n'a qu'un vecteur de logits: un score par token. Il ignore le contexte, donc il ne
 * peut apprendre que "quels tokens sont fréquents globalement". C'est limité, mais parfait pour
 * isoler la mécanique loss -> gradient -> update.
 */
export function createTrainableTokenBiasModel(
    options: TrainableTokenBiasModelOptions,
): TrainableNextTokenModel {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')

    const logits = Array.from({ length: options.vocabularySize }, () => 0)

    return {
        logits,
        predictNextTokenProbabilities: () => softmax(logits),
        vocabularySize: options.vocabularySize,
    }
}

/**
 * Entraîne le mini modèle sur plusieurs epochs.
 *
 * Pour une cross-entropy appliquée après softmax, le gradient par logit est:
 *
 * gradient = probabilitéPrédite - cibleOneHot
 *
 * Version développeur:
 * - si le modèle donne trop de probabilité à un mauvais token, on baisse son logit;
 * - si le modèle ne donne pas assez de probabilité au bon token, on monte son logit;
 * - learningRate contrôle la taille de cette correction.
 */
export function trainNextTokenModel(
    model: TrainableNextTokenModel,
    examples: readonly NextTokenExample[],
    options: TrainingOptions,
): TrainingHistory {
    validateTrainingInputs(model, examples, options)

    const initialLoss = computeAverageLoss(model, examples)
    const epochMetrics: TrainingEpochMetrics[] = []

    for (let epoch = 1; epoch <= options.epochs; epoch++) {
        for (const example of examples) {
            validateTokenId(example.targetTokenId, model.vocabularySize, 'targetTokenId')

            const probabilities = model.predictNextTokenProbabilities()

            for (let tokenId = 0; tokenId < model.vocabularySize; tokenId++) {
                const predictedProbability = readNumberAt(probabilities, tokenId)
                const expectedProbability = tokenId === example.targetTokenId ? 1 : 0
                const currentLogit = readNumberAt(model.logits, tokenId)

                // Descente de gradient:
                // nouveauParamètre = ancienParamètre - learningRate * gradient.
                model.logits[tokenId] =
                    currentLogit -
                    options.learningRate * (predictedProbability - expectedProbability)
            }
        }

        // On mesure la loss après le passage complet sur les exemples.
        // C'est plus lisible pour une démo: chaque ligne montre l'état du modèle après l'epoch.
        const averageLoss = computeAverageLoss(model, examples)

        epochMetrics.push({
            averageLoss,
            epoch,
            perplexity: perplexityFromLoss(averageLoss),
        })
    }

    return {
        epochs: epochMetrics,
        finalLoss: computeAverageLoss(model, examples),
        initialLoss,
    }
}

function computeAverageLoss(
    model: TrainableNextTokenModel,
    examples: readonly NextTokenExample[],
): number {
    const probabilities = model.predictNextTokenProbabilities()
    const lossSum = examples.reduce(
        (sum, example) => sum + crossEntropyLoss(probabilities, example.targetTokenId),
        0,
    )

    return lossSum / examples.length
}

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable à l'index ${String(index)}.`)
    }

    return value
}

function readTokenIdAt(tokenIds: readonly number[], index: number): number {
    const tokenId = tokenIds[index]

    if (tokenId === undefined) {
        throw new Error(`Token introuvable à l'index ${String(index)}.`)
    }

    return tokenId
}

function validateFiniteValues(values: readonly number[], name: string): void {
    for (const [index, value] of values.entries()) {
        if (!Number.isFinite(value)) {
            throw new Error(
                `${name} doit contenir uniquement des nombres finis. Index ${String(
                    index,
                )}: ${String(value)}.`,
            )
        }
    }
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
            `${name} doit être un entier strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validatePositiveNumber(value: number, name: string): void {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(
            `${name} doit être un nombre fini strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validateProbabilityDistributionShape(probabilities: readonly number[]): void {
    if (probabilities.length === 0) {
        throw new Error('probabilities doit contenir au moins une valeur.')
    }

    validateFiniteValues(probabilities, 'probabilities')

    for (const [index, probability] of probabilities.entries()) {
        if (probability < 0) {
            throw new Error(
                `probabilities doit contenir uniquement des valeurs positives ou nulles. Index ${String(
                    index,
                )}: ${String(probability)}.`,
            )
        }
    }
}

function validateTokenId(tokenId: number, vocabularySize: number, name: string): void {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
        throw new Error(
            `${name} doit être un entier entre 0 et ${String(
                vocabularySize - 1,
            )}. Valeur reçue: ${String(tokenId)}.`,
        )
    }
}

function validateTrainingInputs(
    model: TrainableNextTokenModel,
    examples: readonly NextTokenExample[],
    options: TrainingOptions,
): void {
    validatePositiveInteger(model.vocabularySize, 'model.vocabularySize')
    validatePositiveInteger(options.epochs, 'epochs')
    validatePositiveNumber(options.learningRate, 'learningRate')

    if (model.logits.length !== model.vocabularySize) {
        throw new Error(
            `model.logits doit contenir ${String(
                model.vocabularySize,
            )} valeurs. Nombre reçu: ${String(model.logits.length)}.`,
        )
    }

    if (examples.length === 0) {
        throw new Error('trainNextTokenModel attend au moins un exemple.')
    }
}
