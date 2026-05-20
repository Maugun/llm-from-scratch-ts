import {
    compareModelSizes,
    formatBytes,
    formatParameterCount,
    type ModelSizeConfig,
    type ModelSizeEstimate,
} from './index.js'

const configs: readonly ModelSizeConfig[] = [
    {
        contextLength: 4,
        embeddingDimension: 8,
        feedForwardDimension: 16,
        layerCount: 1,
        name: 'Module 14 tiny',
        vocabularySize: 30,
    },
    {
        batchSize: 4,
        contextLength: 128,
        embeddingDimension: 256,
        feedForwardDimension: 1024,
        layerCount: 2,
        name: 'Petit modèle pédagogique',
        vocabularySize: 256,
    },
    {
        batchSize: 4,
        contextLength: 128,
        embeddingDimension: 384,
        feedForwardDimension: 1536,
        layerCount: 4,
        name: 'Cible finale indicative',
        vocabularySize: 512,
    },
]

const estimates = compareModelSizes(configs)

console.info('Module 15 - Estimation mémoire et taille de modèle')
console.info('')
console.info('But du module:')
console.info(
    '  Estimer la taille d’un mini Transformer avant de l’entraîner, pour comprendre les coûts.',
)
console.info('')
console.info('Pourquoi c’est utile:')
console.info('  Un modèle peut sembler petit dans le code, mais ses matrices grandissent vite.')
console.info('  Ce module aide à relier les shapes aux paramètres, à la RAM et à la VRAM.')
console.info('')
console.info('Rappels rapides:')
console.info('  Paramètre: nombre entraînable, par exemple un poids dans une matrice.')
console.info('  float32: format numérique courant, environ 4 bytes par nombre.')
console.info('  Adam: optimizer qui garde deux états en plus par paramètre.')
console.info('  Attention: coût principal en contextLength x contextLength.')
console.info('')
console.info('Comparaison des configurations:')
console.info('')
console.info(
    [
        padRight('Configuration', 28),
        padLeft('Params', 12),
        padLeft('Params RAM', 14),
        padLeft('Train approx.', 16),
        padLeft('Scores attn.', 14),
    ].join('  '),
)

for (const estimate of estimates) {
    console.info(formatSummaryRow(estimate))
}

console.info('')
console.info('Détail de la cible finale indicative:')
printDetailedEstimate(readEstimate(estimates, 2))

console.info('')
console.info('À retenir:')
console.info('  1. Les paramètres contrôlent la mémoire minimale du modèle.')
console.info('  2. Pendant l’entraînement, gradients et états Adam ajoutent beaucoup de mémoire.')
console.info('  3. Le contexte est cher pour l’attention, car le coût grandit au carré.')
console.info(
    '  4. Ces chiffres sont des estimations pédagogiques, pas un profiler TensorFlow.js exact.',
)
console.info('')
console.info('Prochaine étape:')
console.info('  Module 16: introduire @tensorflow/tfjs-node pour préparer un backend plus adapté.')

function formatSummaryRow(estimate: ModelSizeEstimate): string {
    return [
        padRight(estimate.name ?? 'Sans nom', 28),
        padLeft(formatParameterCount(estimate.parameters.total), 12),
        padLeft(formatBytes(estimate.memory.parameterBytes), 14),
        padLeft(formatBytes(estimate.memory.trainingParameterBytes), 16),
        padLeft(formatParameterCount(estimate.attention.scoresTotal), 14),
    ].join('  ')
}

function printDetailedEstimate(estimate: ModelSizeEstimate): void {
    console.info(`  Nom: ${estimate.name ?? 'Sans nom'}`)
    console.info(`  Vocabulaire: ${String(estimate.config.vocabularySize)}`)
    console.info(`  Contexte: ${String(estimate.config.contextLength)}`)
    console.info(`  Embedding dimension: ${String(estimate.config.embeddingDimension)}`)
    console.info(`  Feed-forward dimension: ${String(estimate.config.feedForwardDimension)}`)
    console.info(`  Layers: ${String(estimate.config.layerCount)}`)
    console.info('')
    console.info('  Paramètres:')
    console.info(
        `    embeddings tokens: ${formatParameterCount(estimate.parameters.tokenEmbeddings)}`,
    )
    console.info(
        `    embeddings positions: ${formatParameterCount(estimate.parameters.positionEmbeddings)}`,
    )
    console.info(
        `    attention par layer: ${formatParameterCount(estimate.parameters.attentionPerLayer)}`,
    )
    console.info(
        `    feed-forward par layer: ${formatParameterCount(
            estimate.parameters.feedForwardPerLayer,
        )}`,
    )
    console.info(
        `    blocs Transformer: ${formatParameterCount(
            estimate.parameters.transformerBlocksTotal,
        )}`,
    )
    console.info(
        `    projection vocabulaire: ${formatParameterCount(estimate.parameters.outputProjection)}`,
    )
    console.info(`    total: ${formatParameterCount(estimate.parameters.total)}`)
    console.info('')
    console.info('  Mémoire:')
    console.info(`    paramètres seuls: ${formatBytes(estimate.memory.parameterBytes)}`)
    console.info(`    gradients: ${formatBytes(estimate.memory.gradientBytes)}`)
    console.info(`    états Adam: ${formatBytes(estimate.memory.adamStateBytes)}`)
    console.info(
        `    total entraînement approx.: ${formatBytes(estimate.memory.trainingParameterBytes)}`,
    )
    console.info(
        `    scores attention totaux: ${formatParameterCount(estimate.attention.scoresTotal)}`,
    )
}

function readEstimate(
    estimatesToRead: readonly ModelSizeEstimate[],
    index: number,
): ModelSizeEstimate {
    const estimate = estimatesToRead[index]

    if (estimate === undefined) {
        throw new Error(`Estimation introuvable à l’index ${String(index)}.`)
    }

    return estimate
}

function padLeft(value: string, length: number): string {
    return value.padStart(length, ' ')
}

function padRight(value: string, length: number): string {
    return value.padEnd(length, ' ')
}
