import React, { FC, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@fluentui/react-components'
import { useTranslation } from 'react-i18next'
import { BatchCompletionItem } from '../types/completion'

const cellWidth = 96
const cellGap = 8
const cellStep = cellWidth + cellGap
const overlayHeight = 118
const expandedWidth = 560
const expandedHeight = 360
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
  const pointerXRef = useRef<number | null>(null)
  const [scrollLeft, setScrollLeft] = useState(0)
  const [viewportWidth, setViewportWidth] = useState(800)
  const [hoveredId, setHoveredId] = useState<number | null>(null)
  const [lockedId, setLockedId] = useState<number | null>(null)
  const [activeTextIds, setActiveTextIds] = useState<Set<number>>(new Set())

  const visibleRange = useMemo(() => {
    const start = Math.max(0, Math.floor(scrollLeft / cellStep) - overscan)
    const end = Math.min(
      items.length,
      Math.ceil((scrollLeft + viewportWidth) / cellStep) + overscan
    )
    return { start, end }
  }, [items.length, scrollLeft, viewportWidth])

  const displayRange = useMemo(() => {
    const start = Math.min(
      items.length,
      Math.max(0, Math.floor(scrollLeft / cellStep))
    )
    const end = Math.min(
      items.length,
      Math.max(start, Math.ceil((scrollLeft + viewportWidth) / cellStep))
    )
    return { start, end }
  }, [items.length, scrollLeft, viewportWidth])

  useEffect(() => {
    const current = scrollerRef.current
    if (!current) return

    const updateViewport = () => {
      setViewportWidth(current.clientWidth || 800)
      setScrollLeft(current.scrollLeft)
    }
    updateViewport()

    const observer = new ResizeObserver(updateViewport)
    observer.observe(current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (hoveredId === null) return
    setActiveTextIds((current) => {
      const next = new Set(current)
      next.add(hoveredId)
      return next
    })
  }, [hoveredId])

  if (items.length === 0) return null

  const doneCount = items.filter((item) => item.status === 'done').length
  const activeCount = items.filter(
    (item) => item.status === 'pending' || item.status === 'running'
  ).length
  const updateHoveredIdAt = (clientX: number | null) => {
    if (lockedId !== null) {
      setHoveredId(lockedId)
      return
    }
    const current = scrollerRef.current
    if (!current || clientX === null) {
      setHoveredId(null)
      return
    }
    const rect = current.getBoundingClientRect()
    const x = clientX - rect.left + current.scrollLeft - cellGap
    const index = Math.floor(x / cellStep)
    if (index < 0 || index >= items.length) {
      setHoveredId(null)
      return
    }
    const xInsideCell = x - index * cellStep
    setHoveredId(xInsideCell <= cellWidth ? items[index].id : null)
  }
  const updateHoveredId = (event: React.PointerEvent<HTMLDivElement>) => {
    pointerXRef.current = event.clientX
    updateHoveredIdAt(event.clientX)
  }

  return (
    <div
      className="bg-white/82 absolute inset-x-2 bottom-2 z-10 overflow-visible rounded border border-neutral-200 shadow-lg backdrop-blur"
      style={{ height: overlayHeight }}
    >
      <div className="flex h-8 items-center justify-between border-b border-neutral-200 px-2 text-xs text-neutral-600">
        <span>{t('Batch Generation')}</span>
        <span className="tabular-nums">
          {displayRange.start + 1}-{displayRange.end}/{items.length} ·{' '}
          {doneCount}/{items.length}
          {activeCount > 0 ? ` · ${t('Generating')}` : ''}
        </span>
      </div>
      <div
        className="relative h-[86px] overflow-visible"
        onPointerMove={updateHoveredId}
        onPointerLeave={() => {
          pointerXRef.current = null
          setHoveredId(null)
        }}
        onWheel={(event) => {
          const current = scrollerRef.current
          if (!current) return
          event.preventDefault()
          current.scrollLeft += event.deltaY || event.deltaX
          setScrollLeft(current.scrollLeft)
          updateHoveredIdAt(pointerXRef.current)
        }}
      >
        <div
          ref={scrollerRef}
          className="absolute inset-0 overflow-x-auto overflow-y-hidden"
          onScroll={(event) => {
            setScrollLeft(event.currentTarget.scrollLeft)
            updateHoveredIdAt(pointerXRef.current)
          }}
        >
          <div
            className="h-full"
            style={{ width: items.length * cellStep + cellGap }}
          />
        </div>
        <div
          className="pointer-events-none absolute inset-0 overflow-visible"
          style={{ width: items.length * cellStep + cellGap }}
        >
          {items
            .slice(visibleRange.start, visibleRange.end)
            .map((item, visibleIndex) => {
              const index = visibleRange.start + visibleIndex
              const isHovered = hoveredId === item.id
              const showText = isHovered || activeTextIds.has(item.id)
              const baseLeft = index * cellStep + cellGap - scrollLeft
              const width = isHovered ? expandedWidth : cellWidth
              const height = isHovered ? expandedHeight : 62
              const content = item.error || item.text || ''
              const left = isHovered
                ? Math.min(
                    Math.max(cellGap, baseLeft),
                    Math.max(cellGap, viewportWidth - expandedWidth - cellGap)
                  )
                : baseLeft

              return (
                <div
                  key={item.id}
                  className={
                    'pointer-events-auto absolute overflow-hidden rounded border px-2 py-1 text-xs shadow-sm transition-all duration-150 ease-out ' +
                    statusClass[item.status]
                  }
                  style={{
                    left,
                    top: isHovered ? 50 - expandedHeight : 4,
                    width,
                    height,
                    zIndex: isHovered ? 30 : showText ? 20 : 1,
                  }}
                  onTransitionEnd={(event) => {
                    if (event.target !== event.currentTarget) return
                    if (isHovered) return
                    setActiveTextIds((current) => {
                      if (!current.has(item.id)) return current
                      const next = new Set(current)
                      next.delete(item.id)
                      return next
                    })
                  }}
                  onPointerEnter={() => {
                    if (!isHovered) return
                    setLockedId(item.id)
                    setHoveredId(item.id)
                  }}
                  onPointerLeave={() => {
                    if (lockedId === item.id) setLockedId(null)
                  }}
                >
                  <div className="flex h-5 items-center justify-between gap-1">
                    <span className="font-mono text-[11px]">
                      #{item.id + 1}
                    </span>
                    <div className="flex min-w-0 items-center gap-1">
                      {isHovered && (
                        <Button
                          size="small"
                          appearance="subtle"
                          onClick={(event) => {
                            event.stopPropagation()
                            navigator.clipboard?.writeText(content)
                          }}
                          onPointerDown={(event) => event.stopPropagation()}
                        >
                          {t('Copy')}
                        </Button>
                      )}
                      {item.status === 'running' ? (
                        <Spinner size="tiny" />
                      ) : (
                        <span className="truncate">{t(item.status)}</span>
                      )}
                    </div>
                  </div>
                  {showText ? (
                    <div
                      className="mt-1 h-[calc(100%-24px)] select-text overflow-y-auto whitespace-pre-wrap break-words rounded bg-white/70 px-2 py-1 text-xs leading-5 text-neutral-800"
                      onPointerDown={(event) => {
                        setLockedId(item.id)
                        setHoveredId(item.id)
                        event.stopPropagation()
                      }}
                      onPointerMove={(event) => event.stopPropagation()}
                      onWheel={(event) => event.stopPropagation()}
                    >
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
              )
            })}
        </div>
      </div>
    </div>
  )
}
