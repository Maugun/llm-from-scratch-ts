import { createCharacterTokenizer } from './index.js'

const trainingText = 'bonjour llm.'
const textToEncode = 'llm'
const tokenizer = createCharacterTokenizer(trainingText)
const tokenIds = tokenizer.encode(textToEncode)
const decodedText = tokenizer.decode(tokenIds)

console.info('Module 1 - Tokenizer simple caractere')
console.info('')
console.info('Texte de reference:')
console.info(trainingText)
console.info('')
console.info('Vocabulaire:')
console.info(tokenizer.vocabulary)
console.info('')
console.info(`Texte encode: "${textToEncode}"`)
console.info(tokenIds)
console.info('')
console.info('Texte decode:')
console.info(decodedText)
