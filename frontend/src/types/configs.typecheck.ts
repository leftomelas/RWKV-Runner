import { Device, ModelConfig } from './configs'

const highPerformanceDevice: Device = 'CUDA High Performance'

const highPerformanceConfig: ModelConfig = {
  name: 'Albatross typecheck',
  apiParameters: {
    apiPort: 8000,
    maxResponseToken: 100,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
  },
  modelParameters: {
    modelName: 'rwkv7-test.pth',
    device: highPerformanceDevice,
    precision: 'fp16',
    storedLayers: 0,
    maxStoredLayers: 0,
  },
}

void highPerformanceConfig
