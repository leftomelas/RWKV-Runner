import React, { FC, useEffect, useMemo, useRef, useState } from 'react'
import {
  Button,
  Dropdown,
  Input,
  Option,
  Textarea,
  Tooltip,
} from '@fluentui/react-components'
import { ArrowSync20Regular } from '@fluentui/react-icons'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import {
  StartBatchCompletions,
  StopBatchCompletions,
} from '../../wailsjs/go/backend_golang/App'
import { backend_golang } from '../../wailsjs/go/models'
import { EventsOn } from '../../wailsjs/runtime'
import { BatchCompletionOverlay } from '../components/BatchCompletionOverlay'
import { DialogButton } from '../components/DialogButton'
import { Labeled } from '../components/Labeled'
import { ToolTipButton } from '../components/ToolTipButton'
import { ValuedSlider } from '../components/ValuedSlider'
import { WorkHeader } from '../components/WorkHeader'
import commonStore, { ModelStatus } from '../stores/commonStore'
import {
  BatchCompletionItem,
  BatchCompletionStatus,
  CompletionParams,
  CompletionPreset,
  StopItem,
} from '../types/completion'
import { getReqUrl, getServerRoot, smartScrollHeight } from '../utils'
import { defaultPenaltyDecay, defaultPresets } from './defaultConfigs'
import { PresetsButton } from './PresetsManager/PresetsButton'

let completionSseController: AbortController | null = null

const isNumericStopValue = (value: string) => /^\d+$/.test(value)

const normalizeStopItems = (params: CompletionParams): StopItem[] => {
  if (Array.isArray(params.stopItems)) {
    return params.stopItems
      .map(
        (item) =>
          ({
            type: item?.type === 'token' ? 'token' : 'text',
            value: (item?.value ?? '').toString().trim(),
          }) satisfies StopItem
      )
      .filter((item) => item.value.length > 0)
  }
  const legacyStop = params.stop
  if (typeof legacyStop === 'string' && legacyStop.trim().length > 0) {
    return [{ type: 'text', value: legacyStop.trim() }]
  }
  return []
}

const buildCompletionBody = (
  prompt: string,
  params: CompletionParams,
  stopItems: StopItem[],
  model: string
) => {
  const stopSequences = stopItems
    .filter((item) => item.type === 'text')
    .map((item) => item.value.replaceAll('\\n', '\n'))
    .filter((value) => value.length > 0)
  const stopTokenIds = stopItems
    .filter((item) => item.type === 'token')
    .map((item) => Number.parseInt(item.value, 10))
    .filter((value) => Number.isFinite(value) && value >= 0)

  return {
    prompt,
    stream: true,
    model,
    max_tokens: params.maxResponseToken,
    temperature: params.temperature,
    top_p: params.topP,
    presence_penalty: params.presencePenalty,
    frequency_penalty: params.frequencyPenalty,
    stop: stopSequences.length > 0 ? stopSequences : undefined,
    stop_token_ids: stopTokenIds.length > 0 ? stopTokenIds : undefined,
    penalty_decay:
      !params.penaltyDecay || params.penaltyDecay === defaultPenaltyDecay
        ? undefined
        : params.penaltyDecay,
  }
}

const StopTagInput: FC<{
  items: StopItem[]
  onChange: (items: StopItem[]) => void
  placeholder?: string
}> = ({ items, onChange, placeholder }) => {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLInputElement>(null)
  const [inputValue, setInputValue] = useState('')
  const [focused, setFocused] = useState(false)

  const commitInput = (raw?: string) => {
    const value = (raw ?? inputValue).trim()
    if (!value) return
    const nextItem: StopItem = {
      type: isNumericStopValue(value) ? 'token' : 'text',
      value,
    }
    onChange([...items, nextItem])
    setInputValue('')
  }

  const removeItem = (index: number) => {
    onChange(items.filter((_, i) => i !== index))
  }

  const toggleItem = (index: number) => {
    const item = items[index]
    if (!isNumericStopValue(item.value)) return
    const nextType = item.type === 'token' ? 'text' : 'token'
    onChange(
      items.map((current, i) =>
        i === index ? { ...current, type: nextType } : current
      )
    )
  }

  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      commitInput()
      return
    }
    if (e.key === 'Backspace' && inputValue.length === 0 && items.length > 0) {
      removeItem(items.length - 1)
    }
  }

  const inputWidth = Math.max(6, inputValue.length + 1)

  return (
    <div
      className={
        'flex min-h-[32px] w-full min-w-0 max-w-full flex-wrap items-center gap-1 rounded border px-2 py-1 ' +
        (focused ? 'border-[#115ea3] shadow-sm' : 'border-neutral-300')
      }
      onClick={() => inputRef.current?.focus()}
    >
      {items.map((item, index) => {
        const isNumeric = isNumericStopValue(item.value)
        const isToken = item.type === 'token'
        const title = isToken
          ? isNumeric
            ? t('Stop token id') + ` (${t('Click to toggle type')})`
            : t('Stop token id')
          : isNumeric
            ? t('Stop sequence') + ` (${t('Click to toggle type')})`
            : t('Stop sequence')

        return (
          <Tooltip
            key={`${item.type}-${item.value}-${index}`}
            content={title}
            showDelay={0}
            hideDelay={0}
            relationship="description"
          >
            <span
              className={
                'inline-flex max-w-full items-center gap-1 rounded-full border px-2 py-0.5 text-xs ' +
                (isToken
                  ? 'border-amber-300 bg-amber-100 text-amber-800'
                  : 'border-slate-300 bg-slate-50 text-slate-800') +
                (isNumeric ? ' cursor-pointer' : '')
              }
              onClick={() => toggleItem(index)}
            >
              <span className="break-all">{item.value}</span>
              <div
                className="ml-0.5 inline-flex h-4 w-4 cursor-pointer items-center justify-center rounded-full text-[10px] text-slate-500 hover:bg-slate-200"
                onClick={(e) => {
                  e.stopPropagation()
                  removeItem(index)
                }}
              >
                x
              </div>
            </span>
          </Tooltip>
        )
      })}
      <input
        ref={inputRef}
        className="min-w-[6ch] max-w-full flex-grow bg-transparent text-sm outline-none"
        style={{ width: `${inputWidth}ch` }}
        value={inputValue}
        placeholder={placeholder}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          commitInput()
          setFocused(false)
        }}
      />
    </div>
  )
}

const CompletionPanel: FC = observer(() => {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const batchIdRef = useRef<string | null>(null)
  const batchStartingRef = useRef(false)
  const batchStopRequestedRef = useRef(false)
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort
  const [batchCount, setBatchCount] = useState(16)
  const [batchId, setBatchId] = useState<string | null>(null)
  const [batchStarting, setBatchStarting] = useState(false)
  const [batchItems, setBatchItems] = useState<BatchCompletionItem[]>([])
  const batchAvailable =
    commonStore.platform !== 'web' &&
    typeof (window as any).go?.backend_golang?.App?.StartBatchCompletions ===
      'function'
  const batchActive =
    batchStarting ||
    (batchId !== null &&
      batchItems.some(
        (item) => item.status === 'pending' || item.status === 'running'
      ))

  const scrollToBottom = (force: boolean = false) => {
    const current = inputRef.current
    if (
      current &&
      (force ||
        current.scrollHeight - current.scrollTop - current.clientHeight <
          smartScrollHeight)
    )
      current.scrollTop = current.scrollHeight
  }

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = '100%'
      inputRef.current.style.maxHeight = '100%'
    }
    scrollToBottom(true)
  }, [])

  const setPreset = (preset: CompletionPreset) => {
    commonStore.setCompletionSubmittedPrompt(t(preset.prompt))
    commonStore.setCompletionPreset({
      ...preset,
      prompt: t(preset.prompt),
    })
  }

  if (!commonStore.completionPreset) setPreset(defaultPresets[0])

  const name = commonStore.completionPreset!.name

  const prompt = commonStore.completionPreset!.prompt
  const setPrompt = (prompt: string) => {
    commonStore.setCompletionPreset({
      ...commonStore.completionPreset!,
      prompt,
    })
  }

  const params = commonStore.completionPreset!.params
  const setParams = (newParams: Partial<CompletionParams>) => {
    commonStore.setCompletionPreset({
      ...commonStore.completionPreset!,
      params: {
        ...commonStore.completionPreset!.params,
        ...newParams,
      },
    })
  }

  const stopItems = useMemo(() => normalizeStopItems(params), [params])

  useEffect(() => {
    if (!batchAvailable) return

    const acceptBatchId = (nextBatchId: string) => {
      if (!batchIdRef.current && batchStartingRef.current) {
        batchIdRef.current = nextBatchId
        setBatchId(nextBatchId)
      }
      return batchIdRef.current === nextBatchId
    }

    const unregisterUpdate = EventsOn(
      'batch-completion-update',
      (payload: backend_golang.BatchCompletionUpdateEvent) => {
        if (!payload?.batchId || !acceptBatchId(payload.batchId)) return

        setBatchItems((current) => {
          const next = current.slice()
          for (const update of payload.updates || []) {
            const item = next[update.itemId]
            if (!item) continue
            const status = update.status as BatchCompletionStatus
            next[update.itemId] = {
              ...item,
              status,
              text:
                update.text !== undefined
                  ? update.text
                  : item.text + (update.delta || ''),
              error: update.error || item.error,
            }
          }
          return next
        })
      }
    )

    const unregisterFinished = EventsOn(
      'batch-completion-finished',
      (payload: backend_golang.BatchCompletionFinishedEvent) => {
        if (!payload?.batchId || payload.batchId !== batchIdRef.current) return
        batchIdRef.current = null
        batchStartingRef.current = false
        batchStopRequestedRef.current = false
        setBatchId(null)
        setBatchStarting(false)
      }
    )

    return () => {
      unregisterUpdate()
      unregisterFinished()
      if (batchIdRef.current) {
        StopBatchCompletions(batchIdRef.current).catch(() => {})
      }
    }
  }, [batchAvailable])

  // for old version data compatibility
  useEffect(() => {
    if (!Array.isArray(params.stopItems)) {
      setParams({ stopItems })
    }
  }, [params.stopItems, stopItems])

  const onSubmit = async (prompt: string) => {
    commonStore.setCompletionSubmittedPrompt(prompt)

    if (
      commonStore.status.status === ModelStatus.Offline &&
      !commonStore.settings.apiUrl &&
      commonStore.platform !== 'web'
    ) {
      toast(
        t('Please click the button in the top right corner to start the model'),
        { type: 'warning' }
      )
      commonStore.setCompletionGenerating(false)
      return
    }

    prompt += params.injectStart.replaceAll('\\n', '\n')

    let answer = ''
    let finished = false
    const finish = () => {
      finished = true
      commonStore.setCompletionGenerating(false)
    }
    completionSseController = new AbortController()
    const { url, headers } = await getReqUrl(port, '/v1/completions', true)
    fetchEventSource(
      // https://api.openai.com/v1/completions || http://127.0.0.1:${port}/v1/completions
      url,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`,
          ...headers,
        },
        body: JSON.stringify(
          buildCompletionBody(
            prompt,
            params,
            stopItems,
            commonStore.settings.apiCompletionModelName
          )
        ),
        signal: completionSseController?.signal,
        openWhenHidden: true,
        onmessage(e) {
          if (finished) return
          scrollToBottom()
          if (e.data.trim() === '[DONE]') {
            finish()
            return
          }
          let data
          try {
            data = JSON.parse(e.data)
          } catch (error) {
            console.debug('json error', error)
            return
          }
          if (data.model) commonStore.setLastModelName(data.model)
          if (
            data.choices &&
            Array.isArray(data.choices) &&
            data.choices.length > 0
          ) {
            answer +=
              data.choices[0]?.text || data.choices[0]?.delta?.content || ''
            setPrompt(
              prompt +
                answer.replace(/\s+$/, '') +
                params.injectEnd.replaceAll('\\n', '\n')
            )

            if (data.choices[0]?.finish_reason) {
              finish()
              return
            }
          }
        },
        async onopen(response) {
          if (response.status !== 200) {
            toast(
              response.status +
                ' - ' +
                response.statusText +
                ' - ' +
                (await response.text()),
              {
                type: 'error',
              }
            )
          }
        },
        onclose() {
          console.log('Connection closed')
        },
        onerror(err) {
          err = err.message || err
          if (err && !err.includes('ReadableStreamDefaultReader'))
            toast(err, {
              type: 'error',
            })
          commonStore.setCompletionGenerating(false)
          throw err
        },
      }
    )
  }

  const stopBatch = async () => {
    batchStopRequestedRef.current = true
    const activeBatchId = batchIdRef.current
    batchIdRef.current = null
    batchStartingRef.current = false
    setBatchId(null)
    setBatchStarting(false)
    setBatchItems((current) =>
      current.map((item) =>
        item.status === 'pending' || item.status === 'running'
          ? { ...item, status: 'aborted' }
          : item
      )
    )

    if (activeBatchId) {
      await StopBatchCompletions(activeBatchId).catch(() => {})
    }
  }

  const startBatch = async () => {
    if (batchActive) {
      await stopBatch()
      return
    }
    if (
      commonStore.status.status === ModelStatus.Offline &&
      !commonStore.settings.apiUrl &&
      commonStore.platform !== 'web'
    ) {
      toast(
        t('Please click the button in the top right corner to start the model'),
        { type: 'warning' }
      )
      return
    }

    const count = Math.max(1, Math.min(1000, Math.round(batchCount)))
    setBatchCount(count)
    const requestPrompt = prompt + params.injectStart.replaceAll('\\n', '\n')
    const url = getServerRoot(port, true) + '/v1/completions'
    const headers: Record<string, string> = {}
    if (commonStore.settings.apiKey) {
      headers.Authorization = `Bearer ${commonStore.settings.apiKey}`
    }
    const items = Array.from({ length: count }, (_, id) => ({
      id,
      status: 'pending' as const,
      text: '',
    }))

    batchIdRef.current = null
    batchStartingRef.current = true
    batchStopRequestedRef.current = false
    setBatchItems(items)
    setBatchId(null)
    setBatchStarting(true)

    try {
      const nextBatchId = await StartBatchCompletions({
        url,
        headers,
        count,
        body: buildCompletionBody(
          requestPrompt,
          params,
          stopItems,
          commonStore.settings.apiCompletionModelName
        ),
      })
      batchIdRef.current = nextBatchId
      setBatchId(nextBatchId)
      if (batchStopRequestedRef.current) {
        await stopBatch()
      }
    } catch (error) {
      toast(String(error), { type: 'error' })
      setBatchItems((current) =>
        current.map((item) => ({
          ...item,
          status: 'error',
          error: String(error),
        }))
      )
    } finally {
      batchStartingRef.current = false
      setBatchStarting(false)
    }
  }

  return (
    <div className="flex grow flex-col gap-2 overflow-hidden sm:flex-row">
      <div className="relative min-h-0 grow overflow-hidden">
        <Textarea
          ref={inputRef}
          className="h-full w-full"
          value={prompt}
          onChange={(e) => {
            commonStore.setCompletionSubmittedPrompt(e.target.value)
            setPrompt(e.target.value)
          }}
        />
        {batchAvailable && <BatchCompletionOverlay items={batchItems} />}
      </div>
      <div className="flex max-h-48 flex-col gap-1 sm:max-h-full sm:w-[250px] sm:min-w-[250px] sm:max-w-[250px]">
        <div className="flex gap-2">
          <Dropdown
            style={{ minWidth: 0 }}
            className="grow"
            value={t(commonStore.completionPreset!.name)!}
            selectedOptions={[commonStore.completionPreset!.name]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                setPreset(
                  defaultPresets.find(
                    (preset) => preset.name === data.optionValue
                  )!
                )
              }
            }}
          >
            {defaultPresets.map((preset) => (
              <Option key={preset.name} value={preset.name}>
                {t(preset.name)!}
              </Option>
            ))}
          </Dropdown>
          <PresetsButton tab="Completion" />
        </div>
        <div className="flex flex-col gap-1 overflow-y-auto overflow-x-hidden p-1">
          <Labeled
            flex
            breakline
            label={t('Max Response Token')}
            desc={t(
              'By default, the maximum number of tokens that can be answered in a single response, it can be changed by the user by specifying API parameters.'
            )}
            content={
              <ValuedSlider
                value={params.maxResponseToken}
                min={100}
                max={8100}
                step={100}
                input
                onChange={(e, data) => {
                  setParams({
                    maxResponseToken: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Temperature')}
            desc={t(
              "Sampling temperature, it's like giving alcohol to a model, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be."
            )}
            content={
              <ValuedSlider
                value={params.temperature}
                min={0}
                max={3}
                step={0.1}
                input
                onChange={(e, data) => {
                  setParams({
                    temperature: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Top_P')}
            desc={t(
              'Just like feeding sedatives to the model. Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.'
            )}
            content={
              <ValuedSlider
                value={params.topP}
                min={0}
                max={1}
                step={0.05}
                input
                onChange={(e, data) => {
                  setParams({
                    topP: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Presence Penalty')}
            desc={t(
              "Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."
            )}
            content={
              <ValuedSlider
                value={params.presencePenalty}
                min={0}
                max={2}
                step={0.01}
                input
                onChange={(e, data) => {
                  setParams({
                    presencePenalty: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Frequency Penalty')}
            desc={t(
              "Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."
            )}
            content={
              <ValuedSlider
                value={params.frequencyPenalty}
                min={0}
                max={2}
                step={0.01}
                input
                onChange={(e, data) => {
                  setParams({
                    frequencyPenalty: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={
              t('Penalty Decay') +
              (!params.penaltyDecay ||
              params.penaltyDecay === defaultPenaltyDecay
                ? ` (${t('Default')})`
                : '')
            }
            desc={t("If you don't know what it is, keep it default.")}
            content={
              <ValuedSlider
                value={params.penaltyDecay || defaultPenaltyDecay}
                min={0.99}
                max={0.999}
                step={0.001}
                toFixed={3}
                input
                onChange={(e, data) => {
                  setParams({
                    penaltyDecay: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Stop Sequences')}
            desc={t(
              'When this content appears in the response result, the generation will end.'
            )}
            content={
              <StopTagInput
                items={stopItems}
                onChange={(items) => {
                  setParams({ stopItems: items })
                }}
                placeholder={t('Stop Sequences')!}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Inject start text')}
            desc={t('Before the response starts, inject this content.')}
            content={
              <Input
                value={params.injectStart}
                onChange={(e, data) => {
                  setParams({
                    injectStart: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Inject end text')}
            desc={t('When response finished, inject this content.')}
            content={
              <Input
                value={params.injectEnd}
                onChange={(e, data) => {
                  setParams({
                    injectEnd: data.value,
                  })
                }}
              />
            }
          />
        </div>
        <div className="grow" />
        {batchAvailable && (
          <Labeled
            flex
            breakline
            label={t('Batch Count')}
            desc={t('Number of concurrent completion requests to start.')}
            content={
              <Input
                type="number"
                min={1}
                max={1000}
                value={String(batchCount)}
                onChange={(_, data) => {
                  const value = Number.parseInt(data.value, 10)
                  setBatchCount(Number.isFinite(value) ? value : 1)
                }}
              />
            }
          />
        )}
        <div className="hidden justify-between gap-2 sm:flex">
          <Button
            className="grow"
            onClick={() => {
              const newPrompt = prompt
                .replace(/\n+\ /g, '\n')
                .split('\n')
                .map((line) => line.trim())
                .join('\n')
              setPrompt(newPrompt)
              commonStore.setCompletionSubmittedPrompt(newPrompt)
            }}
          >
            {t('Format Content')}
          </Button>
        </div>
        <div className="flex justify-between gap-2">
          <ToolTipButton
            desc={t('Regenerate')}
            icon={<ArrowSync20Regular />}
            onClick={() => {
              completionSseController?.abort()
              commonStore.setCompletionGenerating(true)
              setPrompt(commonStore.completionSubmittedPrompt)
              onSubmit(commonStore.completionSubmittedPrompt)
            }}
          />
          <DialogButton
            className="grow"
            text={t('Reset')}
            title={t('Reset')}
            content={t(
              'Are you sure you want to reset this page? It cannot be undone.'
            )}
            onConfirm={() => {
              setPreset(defaultPresets.find((preset) => preset.name === name)!)
            }}
          />
          <Button
            className="grow"
            appearance="primary"
            onClick={() => {
              if (commonStore.completionGenerating) {
                completionSseController?.abort()
                commonStore.setCompletionGenerating(false)
              } else {
                commonStore.setCompletionGenerating(true)
                onSubmit(prompt)
              }
            }}
          >
            {!commonStore.completionGenerating ? t('Generate') : t('Stop')}
          </Button>
        </div>
        {batchAvailable && (
          <Button
            appearance={batchActive ? 'secondary' : 'primary'}
            onClick={startBatch}
          >
            {batchActive
              ? t('Stop Batch Generation')
              : t('Start Batch Generation')}
          </Button>
        )}
      </div>
    </div>
  )
})

const Completion: FC = observer(() => {
  return (
    <div className="flex h-full flex-col gap-1 overflow-hidden p-2">
      <WorkHeader />
      <CompletionPanel />
    </div>
  )
})

export default Completion
