import { createCharacterTokenizer } from './index.js'

const trainingText = 'bonjour llm.'
const textToEncode = 'llm'
const tokenizer = createCharacterTokenizer(trainingText)
const tokenIds = tokenizer.encode(textToEncode)
const decodedText = tokenizer.decode(tokenIds)

console.info('Module 1 - Tokenizer simple caractère')
console.info('')
console.info('Texte de référence:')
console.info(trainingText)
console.info('')
console.info('Vocabulaire:')
console.info(tokenizer.vocabulary)
console.info('')
console.info(`Texte encodé: "${textToEncode}"`)
console.info(tokenIds)
console.info('')
console.info('Texte décodé:')
console.info(decodedText)
