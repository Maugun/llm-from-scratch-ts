import {
    benchmarkTfjsNodeGpuMatMul,
    getRuntimeEnvironmentInfo,
    loadTfjsNodeGpuBackend,
    runTfjsNodeGpuSmokeTest,
} from './index.js'

const benchmarkMatrixSize = 128
const benchmarkIterations = 5

console.info('Module 16 - Backend GPU TensorFlow.js sous WSL2, Linux et macOS')
console.info('')
console.info('But du module:')
console.info(
    '  Vérifier si TensorFlow.js peut utiliser un backend GPU CUDA via @tensorflow/tfjs-node-gpu.',
)
console.info('')
console.info('Pourquoi c’est important:')
console.info('  @tensorflow/tfjs pur JS est parfait pour apprendre, mais trop lent pour viser')
console.info('  confortablement un petit LLM sur un corpus plus long.')
console.info('  Le backend GPU doit permettre d’exploiter la VRAM d’une carte NVIDIA compatible.')
console.info('')
console.info('Infos runtime:')

const runtime = getRuntimeEnvironmentInfo()

console.info(`  Node: ${runtime.nodeVersion}`)
console.info(`  Plateforme: ${runtime.platform}`)
console.info(`  Architecture: ${runtime.architecture}`)
console.info(`  OS: ${runtime.osType} ${runtime.osRelease}`)
console.info(`  Backend TensorFlow.js actuel: ${runtime.currentBackend}`)
console.info(`  Version TensorFlow.js: ${runtime.tfjsVersion}`)
console.info('')
console.info('Chargement du backend GPU:')
console.info('  TensorFlow peut afficher des logs natifs avant la suite de la démo.')
console.info('  Sous WSL, les messages NUMA sont fréquents et généralement non bloquants.')
console.info(
    '  La ligne importante à chercher est: Created device ... GPU:0 avec une quantité de mémoire.',
)
console.info('')

const backend = await loadTfjsNodeGpuBackend()

if (!backend.available) {
    console.info('Backend GPU non disponible dans cet environnement.')
    console.info(`Erreur: ${backend.errorMessage}`)
    console.info('')
    console.info('Pistes à vérifier:')

    for (const item of backend.guidance) {
        console.info(`  - ${item}`)
    }

    console.info('')
    console.info('Compatibilité rapide:')
    console.info('  Windows + NVIDIA: passer par WSL2 Ubuntu, puis vérifier nvidia-smi dans WSL.')
    console.info('  Linux + NVIDIA: installer driver NVIDIA, CUDA, cuDNN et Node LTS.')
    console.info('  macOS: pas de CUDA, donc pas de @tensorflow/tfjs-node-gpu.')
    console.info('')
    console.info('La démo s’arrête proprement pour garder le projet utilisable sans GPU.')
    process.exit(0)
}

console.info('Backend GPU chargé:')
console.info(`  Backend: ${backend.backendName}`)
console.info(`  Version TensorFlow.js: ${backend.tfjsVersion}`)
console.info(`  Tenseurs actifs au chargement: ${String(backend.tensorCount)}`)
console.info('')

const smokeTest = await runTfjsNodeGpuSmokeTest()

if (!smokeTest.available) {
    throw new Error(smokeTest.errorMessage)
}

console.info('Smoke test matMul:')
console.info('  Entrée: [[1, 2], [3, 4]] x [[5, 6], [7, 8]]')
console.info(`  Attendu: ${JSON.stringify(smokeTest.expected)}`)
console.info(`  Obtenu:  ${JSON.stringify(smokeTest.actual)}`)
console.info(`  Tenseurs avant: ${String(smokeTest.tensorCountBefore)}`)
console.info(`  Tenseurs après: ${String(smokeTest.tensorCountAfter)}`)
console.info('')

const benchmark = await benchmarkTfjsNodeGpuMatMul({
    iterations: benchmarkIterations,
    matrixSize: benchmarkMatrixSize,
})

if (!benchmark.available) {
    throw new Error(benchmark.errorMessage)
}

console.info('Micro-benchmark indicatif:')
console.info(`  Matrices: ${String(benchmark.matrixSize)} x ${String(benchmark.matrixSize)}`)
console.info(`  Itérations: ${String(benchmark.iterations)}`)
console.info(`  Durée totale: ${benchmark.durationMs.toFixed(2)} ms`)
console.info(`  Moyenne par itération: ${benchmark.averageIterationMs.toFixed(2)} ms`)
console.info('')
console.info('À retenir:')
console.info('  1. Le backend GPU est un moteur d’exécution, pas un nouveau modèle.')
console.info('  2. CUDA/cuDNN rendent possible l’usage de la VRAM NVIDIA.')
console.info('  3. Le vrai gain apparaîtra surtout sur des tenseurs et batchs plus grands.')
console.info('  4. Les modules 17 et 18 pourront s’appuyer sur ce socle si le GPU est disponible.')
