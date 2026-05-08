import React, { FC, useMemo, useRef, useState } from 'react'
import { Spinner, Tooltip } from '@fluentui/react-components'
import { useTranslation } from 'react-i18next'
import { BatchCompletionItem } from '../types/completion'

const cellWidth = 96
const cellGap = 8
const cellStep = cellWidth + cellGap
const overlayHeight = 118
const expandedWidth = 286
const expandedHeight = 154
const overscan = 6

const statusClass: Record<BatchCompletionItem['status'], string> = {
  pending: 'border-neutral-300 bg-white/92 text-neutral-500',
  running: 'border-sky-300 bg-sky-50/95 text-sky-700',
  done: 'border-emerald-300 bg-emerald-50/95 text-emerald-800',
  error: 'border-red-300 bg-red-50/95 text-red-800',
  aborted: 'border-neutral-300 bg-neutral-100/95 text-neutral-500',
}

export const BatchCompletionOverlay: FC<{
  items: BatchCompletionItem[]
}> = ({ items }) => {
  const { t } = useTranslation()
  const scrollerRef = useRef<HTMLDivElement>(null)
  const [scrollLeft, setScrollLeft] = useState(0)
  const [hoveredId, setHoveredId] = useState<number | null>(null)

  const visibleRange = useMemo(() => {
    const scrollerWidth = scrollerRef.current?.clientWidth ?? 800
    const start = Math.max(0, Math.floor(scrollLeft / cellStep) - overscan)
    const end = Math.min(
      items.length,
      Math.ceil((scrollLeft + scrollerWidth) / cellStep) + overscan
    )
    return { start, end }
  }, [items.length, scrollLeft])

  const textRenderIds = useMemo(() => {
    if (hoveredId === null) return new Set<number>()
    return new Set([hoveredId - 1, hoveredId, hoveredId + 1])
  }, [hoveredId])

  if (items.length === 0) return null

  const doneCount = items.filter((item) => item.status === 'done').length
  const activeCount = items.filter(
    (item) => item.status === 'pending' || item.status === 'running'
  ).length

  return (
    <div
      className="bg-white/82 absolute inset-x-2 bottom-2 z-10 rounded border border-neutral-200 shadow-lg backdrop-blur"
      style={{ height: overlayHeight }}
    >
      <div className="flex h-8 items-center justify-between border-b border-neutral-200 px-2 text-xs text-neutral-600">
        <span>{t('Batch Generation')}</span>
        <span>
          {doneCount}/{items.length}
          {activeCount > 0 ? ` · ${t('Generating')}` : ''}
        </span>
      </div>
      <div
        ref={scrollerRef}
        className="relative h-[86px] overflow-hidden"
        onScroll={(event) => setScrollLeft(event.currentTarget.scrollLeft)}
        onWheel={(event) => {
          const current = scrollerRef.current
          if (!current) return
          event.preventDefault()
          current.scrollLeft += event.deltaY || event.deltaX
          setScrollLeft(current.scrollLeft)
        }}
      >
        <div
          className="relative h-full"
          style={{ width: items.length * cellStep + cellGap }}
        >
          {items
            .slice(visibleRange.start, visibleRange.end)
            .map((item, visibleIndex) => {
              const index = visibleRange.start + visibleIndex
              const isHovered = hoveredId === item.id
              const showText = textRenderIds.has(item.id)
              const left = index * cellStep + cellGap
              const width = isHovered ? expandedWidth : cellWidth
              const height = isHovered ? expandedHeight : 62

              return (
                <Tooltip
                  key={item.id}
                  content={`#${item.id + 1} · ${t(item.status)}`}
                  relationship="description"
                >
                  <div
                    className={
                      'absolute bottom-3 overflow-hidden rounded border px-2 py-1 text-xs shadow-sm transition-all duration-150 ease-out ' +
                      statusClass[item.status]
                    }
                    style={{
                      left,
                      width,
                      height,
                      transform: isHovered ? 'translateY(-58px)' : undefined,
                      zIndex: isHovered ? 30 : showText ? 20 : 1,
                    }}
                    onMouseEnter={() => setHoveredId(item.id)}
                    onMouseLeave={() => setHoveredId(null)}
                  >
                    <div className="flex h-5 items-center justify-between gap-1">
                      <span className="font-mono text-[11px]">
                        #{item.id + 1}
                      </span>
                      {item.status === 'running' ? (
                        <Spinner size="tiny" />
                      ) : (
                        <span className="truncate">{t(item.status)}</span>
                      )}
                    </div>
                    {showText ? (
                      <div className="mt-1 h-[calc(100%-24px)] select-text overflow-y-auto whitespace-pre-wrap break-words rounded bg-white/70 px-1 py-0.5 text-[11px] leading-4 text-neutral-800">
                        {item.error || item.text || t('Generating')}
                      </div>
                    ) : (
                      <div className="mt-2 grid grid-cols-4 gap-1">
                        {Array.from({ length: 12 }).map((_, line) => (
                          <span
                            key={line}
                            className={
                              'h-1 rounded-full ' +
                              (item.status === 'done'
                                ? 'bg-emerald-300/70'
                                : item.status === 'error'
                                  ? 'bg-red-300/70'
                                  : 'bg-neutral-300/80')
                            }
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </Tooltip>
              )
            })}
        </div>
      </div>
    </div>
  )
}
