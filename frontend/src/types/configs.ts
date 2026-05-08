export type ApiParameters = {
  apiPort: number
  maxResponseToken: number
  temperature: number
  topP: number
  presencePenalty: number
  frequencyPenalty: number
  penaltyDecay?: number
  globalPenalty?: boolean
  stateModel?: string
}
export const defaultAlbatrossWorkers = 1
export const defaultAlbatrossBatch = 960

export type Device =
  | 'CPU'
  | 'CPU (rwkv.cpp)'
  | 'CUDA'
  | 'CUDA-Beta'
  | 'CUDA High Performance'
  | 'WebGPU'
  | 'WebGPU (Python)'
  | 'MPS'
  | 'Custom'
export type Precision = 'fp16' | 'int8' | 'fp32' | 'nf4' | 'Q5_1'
export type GGUFMode = 'CPU' | 'Vulkan GPU'
export type ModelParameters = {
  // different models can not have the same name
  modelName: string
  device: Device
  precision: Precision
  storedLayers: number
  maxStoredLayers: number
  quantizedLayers?: number
  tokenChunkSize?: number
  useCustomCuda?: boolean
  customStrategy?: string
  albatrossWorkers?: number
  albatrossBatch?: number
  useCustomTokenizer?: boolean
  customTokenizer?: string
  ggufMode?: GGUFMode
  llamaContext?: number
}
export type ModelConfig = {
  // different configs can have the same name
  name: string
  apiParameters: ApiParameters
  modelParameters: ModelParameters
  enableWebUI?: boolean
}
