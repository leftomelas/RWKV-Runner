import {
  AddToDownloadList,
  ChangeFileLine,
  FileExists,
  GetAbsPath,
  GetPython,
  ListDirFiles,
  PauseDownload,
  ReadFileInfo,
  SaveFile,
} from '../../wailsjs/go/backend_golang/App'
import { EventsOn } from '../../wailsjs/runtime/runtime'
import { OutputHandler } from '../stores/cmdTaskChainStore'
import commonStore from '../stores/commonStore'
import { DownloadStatus } from '../types/downloads'
import { cmdInteractive, readFile } from './index'

export interface TaskResult {
  /** may be changed dynamically */
  stop: () => void
  /** only cmdInteractive has eventId */
  eventId?: string
  /** if the task is stopped by an explicit action, return false, otherwise return true */
  promise: Promise<boolean>
}

export const ImmediateTaskResult: TaskResult = {
  stop: () => {},
  promise: Promise.resolve(true),
}

function createTaskTwoToOne(
  taskResult: TaskResult,
  cmdArgs: string[],
  onOutput?: OutputHandler
): TaskResult {
  taskResult.promise.then(() => {
    const task2 = createTask(cmdArgs, onOutput)
    taskResult.stop = task2.stop
    taskResult.eventId = task2.eventId
    return task2.promise
  })
  return taskResult
}

export function createTaskMultiToOne(
  cmdArgs: string[][],
  onOutput?: OutputHandler
): TaskResult {
  if (cmdArgs.length < 2) {
    throw new Error('length of cmdArgs must be at least 2')
  }

  let combinedTask = createTask(cmdArgs[0], onOutput)
  for (let i = 1; i < cmdArgs.length; i++) {
    combinedTask = createTaskTwoToOne(combinedTask, cmdArgs[i], onOutput)
  }
  return combinedTask
}

export function createTask(
  cmdArgs: string[],
  onOutput?: OutputHandler
): TaskResult {
  let resolve: (stopped: boolean) => void
  let reject: (reason?: any) => void
  const promise = new Promise<boolean>((r, j) => {
    resolve = r
    reject = j
  })
  const result = cmdInteractive(
    cmdArgs,
    async (output: string) => {
      onOutput?.(output)
    },
    async (stopped: boolean) => {
      resolve(!stopped)
    },
    async (error: string) => {
      reject(new Error(error))
    }
  )

  return {
    stop: result.stop,
    eventId: result.eventId,
    promise,
  }
}

export async function addToDownloadList(
  path: string,
  url: string,
  onOutput: OutputHandler
): Promise<TaskResult> {
  let resolve: (stopped: boolean) => void
  let reject: (reason?: any) => void
  const promise = new Promise<boolean>((r, j) => {
    resolve = r
    reject = j
  })

  const absPath = await GetAbsPath(path)
  const cancel = EventsOn(
    'downloadList',
    (data: DownloadStatus[] | undefined) => {
      if (data) {
        const status = data.find((status) => status.path === absPath)
        if (status) {
          if (status.done) {
            cancel()
            resolve(true)
            onOutput(
              'done: ' +
                status.name +
                ': ' +
                status.progress.toFixed(2) +
                '% ' +
                JSON.stringify(status)
            )
          } else if (!status.downloading) {
            cancel()
            resolve(false)
            onOutput(
              'stopped: ' +
                status.name +
                ': ' +
                status.progress.toFixed(2) +
                '% ' +
                JSON.stringify(status)
            )
          } else {
            onOutput(
              status.name +
                ': ' +
                status.progress.toFixed(2) +
                '% ' +
                JSON.stringify(status)
            )
          }
        }
      }
    }
  )

  const stop = () => {
    cancel()
    PauseDownload(url)
    resolve(false)
  }

  AddToDownloadList(path, url)
  return { stop, promise }
}

export async function startServer(
  python: string,
  port: number,
  host: string,
  webui: boolean,
  rwkvBeta: boolean,
  rwkvcpp: boolean,
  webgpu: boolean,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './backend-python/main.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('main.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const args = [python, execFile]
  if (webui) args.push('--webui')
  // if (rwkvBeta) args.push('--rwkv-beta')
  if (rwkvcpp) args.push('--rwkv.cpp')
  if (webgpu) args.push('--webgpu')
  args.push('--port', port.toString(), '--host', host)

  return createTask(args, onOutput)
}

export async function startWebGPUServer(
  port: number,
  host: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  let execFile = ''
  const execFiles = [
    './backend-rust/webgpu_server',
    './backend-rust/webgpu_server.exe',
  ]
  for (const file of execFiles) {
    const exists = await FileExists(file)
    if (exists) {
      execFile = file
      break
    }
  }

  if (!execFile) {
    throw new Error(execFiles[0] + ' not found')
  }

  const args = [execFile, '--port', port.toString(), '--ip', host]

  return createTask(args, onOutput)
}

export async function convertModel(
  python: string,
  modelPath: string,
  strategy: string,
  outPath: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './backend-python/convert_model.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('convert_model.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const args = [
    python,
    execFile,
    '--in',
    modelPath,
    '--out',
    outPath,
    '--strategy',
    strategy,
  ]

  return createTask(args, onOutput)
}

export async function convertSafetensors(
  modelPath: string,
  outPath: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  let execFile = ''
  const execFiles = [
    './backend-rust/web-rwkv-converter',
    './backend-rust/web-rwkv-converter.exe',
  ]
  for (const file of execFiles) {
    const exists = await FileExists(file)
    if (exists) {
      execFile = file
      break
    }
  }

  if (!execFile) {
    throw new Error(execFiles[0] + ' not found')
  }

  const args = [execFile, '--input', modelPath, '--output', outPath]

  return createTask(args, onOutput)
}

export async function convertSafetensorsWithPython(
  python: string,
  modelPath: string,
  outPath: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './backend-python/convert_safetensors.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('convert_safetensors.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const args = [python, execFile, '--input', modelPath, '--output', outPath]

  return createTask(args, onOutput)
}

export async function convertGGML(
  python: string,
  modelPath: string,
  outPath: string,
  Q51: boolean,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './backend-python/convert_pytorch_to_ggml.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('convert_pytorch_to_ggml.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const dataType = Q51 ? 'Q5_1' : 'FP16'
  const args = [python, execFile, modelPath, outPath, dataType]

  return createTask(args, onOutput)
}

export async function convertData(
  python: string,
  input: string,
  outputPrefix: string,
  vocab: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './finetune/json2binidx_tool/tools/preprocess_data.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('preprocess_data.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const tokenizerType = vocab.includes('rwkv_vocab_v20230424')
    ? 'RWKVTokenizer'
    : 'HFTokenizer'

  input = input.replace(/\/$/, '')
  const inputInfo = await ReadFileInfo(input)
  console.log('input size', inputInfo.size)
  if (inputInfo.isDir) {
    const files = await ListDirFiles(input)
    let jsonlContent = ''
    for (const file of files) {
      if (file.isDir || !file.name.endsWith('.txt')) {
        continue
      }
      const textContent = await (await readFile(input + '/' + file.name)).text()
      const textJson = JSON.stringify({
        text: textContent.replaceAll('\r\n', '\n').replaceAll('\r', '\n'),
      })
      jsonlContent += textJson + '\n'
    }
    await SaveFile(
      outputPrefix + '.jsonl',
      Array.from(new TextEncoder().encode(jsonlContent))
    )
    input = outputPrefix + '.jsonl'
  }

  const args = [
    python,
    execFile,
    '--input',
    input,
    '--output-prefix',
    outputPrefix,
    '--vocab',
    vocab,
    '--tokenizer-type',
    tokenizerType,
    '--dataset-impl',
    'mmap',
    '--append-eod',
  ]

  return createTask(args, onOutput)
}

export async function mergeLora(
  python: string,
  useGpu: boolean,
  loraAlpha: number,
  baseModel: string,
  loraPath: string,
  outputPath: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './finetune/lora/merge_lora.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('merge_lora.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const args = [python, execFile]
  if (useGpu) args.push('--use-gpu')
  args.push(loraAlpha.toString(), baseModel, loraPath, outputPath)

  return createTask(args, onOutput)
}

export async function depCheck(
  python: string,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  const execFile = './backend-python/dep_check.py'
  const exists = await FileExists(execFile)
  if (!exists) {
    throw new Error('dep_check.py not found')
  }
  if (python === '') {
    python = await GetPython()
  }
  const args = [python, execFile]

  return createTask(args, onOutput)
}

export async function installPyDep(
  python: string,
  cnMirror: boolean,
  onOutput?: OutputHandler
): Promise<TaskResult> {
  let torchWhlUrl =
    'torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117'
  if (python === '') {
    python = await GetPython()
    if (cnMirror && python === 'py310/python.exe') {
      torchWhlUrl =
        'https://mirrors.aliyun.com/pytorch-wheels/cu117/torch-1.13.1+cu117-cp310-cp310-win_amd64.whl'
    }
  }

  if (commonStore.platform === 'windows') {
    ChangeFileLine('./py310/python310._pth', 3, 'Lib\\site-packages')
    let installScript =
      python +
      ' ./backend-python/get-pip.py -i https://mirrors.aliyun.com/pypi/simple --no-warn-script-location\n' +
      python +
      ' -m pip install ' +
      torchWhlUrl +
      ' --no-warn-script-location\n' +
      python +
      ' -m pip install -r ./backend-python/requirements.txt -i https://mirrors.aliyun.com/pypi/simple --no-warn-script-location'
    if (!cnMirror) {
      installScript = installScript.replaceAll(
        ' -i https://mirrors.aliyun.com/pypi/simple',
        ''
      )
    }

    const installSteps = installScript
      .split('\n')
      .map((step) => step.split(' '))
    return createTaskMultiToOne(installSteps, onOutput)
  }

  const args = [
    python,
    '-m',
    'pip',
    'install',
    '-r',
    './backend-python/requirements_without_cyac.txt',
  ]
  if (cnMirror) {
    args.push('-i', 'https://mirrors.aliyun.com/pypi/simple')
  }

  return createTask(args, onOutput)
}
