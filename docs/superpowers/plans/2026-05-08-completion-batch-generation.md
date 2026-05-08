# Completion Batch Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a desktop-only batch generation demo on the Completion page that sends up to 1000 concurrent completion requests through Go and displays results in an animated virtual overlay.

**Architecture:** Go owns request fan-out, streaming SSE parsing, cancellation, and Wails event batching. React owns local UI state, virtualized result rendering, hover animation, and desktop-only presentation. The normal Completion textarea and Generate flow remain unchanged.

**Tech Stack:** Go 1.22, Wails v2 runtime events, React, TypeScript, Fluent UI, Tailwind utility classes.

---

## File Structure

- Create `backend-golang/batch_completion.go`: Wails methods, active batch registry, SSE parsing, event aggregation.
- Create `backend-golang/batch_completion_test.go`: Go unit tests with `httptest.Server`.
- Modify generated Wails bindings by running `wails generate`: `frontend/wailsjs/go/backend_golang/App.d.ts`, `frontend/wailsjs/go/backend_golang/App.js`, and `frontend/wailsjs/go/models.ts`.
- Create `frontend/src/pages/CompletionBatch/types.ts`: frontend batch types.
- Create `frontend/src/pages/CompletionBatch/virtualGrid.ts`: pure virtual range helpers.
- Create `frontend/src/pages/CompletionBatch/batchCompletionClient.ts`: Wails method calls and event subscription wrapper.
- Create `frontend/src/pages/CompletionBatch/BatchCompletionOverlay.tsx`: controls, reducer, overlay, virtual cells, hover text rendering.
- Create `frontend/src/pages/CompletionBatch/virtualGrid.test.ts`: pure helper tests if the repo test runner is available; otherwise keep helpers simple and verify with TypeScript.
- Modify `frontend/src/pages/Completion.tsx`: wrap textarea in a relative container and mount the desktop-only overlay.
- Modify locale files if needed: `frontend/src/_locales/zh-hans/main.json`, `frontend/src/_locales/ja/main.json`.

---

### Task 1: Go Batch Completion Service Tests

**Files:**
- Create: `backend-golang/batch_completion_test.go`
- Create in Task 2: `backend-golang/batch_completion.go`

- [ ] **Step 1: Write failing validation and SSE parsing tests**

Create `backend-golang/batch_completion_test.go` with:

```go
package backend_golang

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestBatchCompletionRequestValidation(t *testing.T) {
	app := NewApp()
	_, err := app.StartBatchCompletions(BatchCompletionRequest{
		URL:   "",
		Count: 1,
		Body:  map[string]any{"prompt": "hello"},
	})
	if err == nil || !strings.Contains(err.Error(), "url is required") {
		t.Fatalf("expected url validation error, got %v", err)
	}

	_, err = app.StartBatchCompletions(BatchCompletionRequest{
		URL:   "http://127.0.0.1:1234/v1/completions",
		Count: 1001,
		Body:  map[string]any{"prompt": "hello"},
	})
	if err == nil || !strings.Contains(err.Error(), "count must be between 1 and 1000") {
		t.Fatalf("expected count validation error, got %v", err)
	}
}

func TestParseCompletionSSEText(t *testing.T) {
	line := []byte(`data: {"choices":[{"text":"hello"}]}`)
	delta, done, err := parseCompletionSSELine(line)
	if err != nil {
		t.Fatal(err)
	}
	if done {
		t.Fatal("did not expect done")
	}
	if delta != "hello" {
		t.Fatalf("delta = %q", delta)
	}

	delta, done, err = parseCompletionSSELine([]byte(`data: [DONE]`))
	if err != nil {
		t.Fatal(err)
	}
	if !done || delta != "" {
		t.Fatalf("expected done with empty delta, got done=%v delta=%q", done, delta)
	}
}

func TestBatchCompletionFanoutEmitsUpdates(t *testing.T) {
	requests := make(chan struct{}, 8)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- struct{}{}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"text\":\"hello\"}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer server.Close()

	app := NewApp()
	app.batchCompletionEmit = func(event string, data any) {
		if event != batchCompletionUpdateEvent {
			return
		}
		payload := data.(BatchCompletionUpdateEvent)
		if len(payload.Updates) == 0 {
			t.Fatal("expected at least one update")
		}
	}

	batchID, err := app.StartBatchCompletions(BatchCompletionRequest{
		URL:   server.URL,
		Count: 2,
		Body:  map[string]any{"prompt": "hello", "stream": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if batchID == "" {
		t.Fatal("expected batch id")
	}

	for i := 0; i < 2; i++ {
		select {
		case <-requests:
		case <-time.After(2 * time.Second):
			t.Fatalf("request %d was not sent", i)
		}
	}
}

func TestBatchCompletionStopCancelsActiveRun(t *testing.T) {
	block := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	defer server.Close()
	defer close(block)

	app := NewApp()
	batchID, err := app.StartBatchCompletions(BatchCompletionRequest{
		URL:   server.URL,
		Count: 1,
		Body:  map[string]any{"prompt": "hello", "stream": true},
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := app.StopBatchCompletions(batchID); err != nil {
		t.Fatal(err)
	}
	if err := app.StopBatchCompletions(batchID); err == nil {
		t.Fatal("expected unknown batch error after first stop")
	}
}

func decodeSSEForTest(t *testing.T, body string) []string {
	t.Helper()
	scanner := bufio.NewScanner(strings.NewReader(body))
	var deltas []string
	for scanner.Scan() {
		delta, done, err := parseCompletionSSELine(scanner.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		if done {
			break
		}
		if delta != "" {
			deltas = append(deltas, delta)
		}
	}
	return deltas
}

func TestDecodeSSEForTestHelper(t *testing.T) {
	encoded, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"delta": map[string]any{"content": "x"}}},
	})
	deltas := decodeSSEForTest(t, "data: "+string(encoded)+"\n\ndata: [DONE]\n\n")
	if len(deltas) != 1 || deltas[0] != "x" {
		t.Fatalf("unexpected deltas: %#v", deltas)
	}
}

var _ = context.Background
```

- [ ] **Step 2: Run tests and verify they fail because production code is missing**

Run:

```powershell
go test ./backend-golang -run BatchCompletion
```

Expected: compile failure naming missing `BatchCompletionRequest`, `StartBatchCompletions`, `StopBatchCompletions`, and `parseCompletionSSELine`.

- [ ] **Step 3: Commit only the failing tests if following strict TDD checkpoints**

```powershell
git add backend-golang/batch_completion_test.go
git commit -m "test: define batch completion backend contract"
```

---

### Task 2: Implement Go Batch Completion Service

**Files:**
- Create: `backend-golang/batch_completion.go`
- Modify: `backend-golang/batch_completion_test.go` only if compile details require small alignment

- [ ] **Step 1: Add Go service implementation**

Create `backend-golang/batch_completion.go`:

```go
package backend_golang

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

const (
	batchCompletionUpdateEvent   = "batch-completion-update"
	batchCompletionFinishedEvent = "batch-completion-finished"
)

type BatchCompletionRequest struct {
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Count   int               `json:"count"`
	Body    map[string]any    `json:"body"`
}

type BatchCompletionUpdateEvent struct {
	BatchID string                  `json:"batchId"`
	Updates []BatchCompletionUpdate `json:"updates"`
}

type BatchCompletionFinishedEvent struct {
	BatchID string `json:"batchId"`
}

type BatchCompletionUpdate struct {
	ItemID int    `json:"itemId"`
	Status string `json:"status"`
	Delta  string `json:"delta,omitempty"`
	Text   string `json:"text,omitempty"`
	Error  string `json:"error,omitempty"`
}

type batchCompletionRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

type batchCompletionEmitFunc func(event string, data any)

var batchCompletionHTTPClient = &http.Client{Timeout: 0}

func (a *App) ensureBatchCompletionState() {
	if a.batchCompletionRuns == nil {
		a.batchCompletionRuns = map[string]*batchCompletionRun{}
	}
	if a.batchCompletionEmit == nil {
		a.batchCompletionEmit = func(event string, data any) {
			if a.ctx != nil {
				wruntime.EventsEmit(a.ctx, event, data)
			}
		}
	}
}

func (a *App) StartBatchCompletions(req BatchCompletionRequest) (string, error) {
	if err := req.validate(); err != nil {
		return "", err
	}

	a.ensureBatchCompletionState()
	batchID := newBatchCompletionID()
	ctx, cancel := context.WithCancel(context.Background())
	run := &batchCompletionRun{cancel: cancel, done: make(chan struct{})}

	a.batchCompletionMu.Lock()
	a.batchCompletionRuns[batchID] = run
	a.batchCompletionMu.Unlock()

	updates := make(chan BatchCompletionUpdate, 4096)
	var wg sync.WaitGroup
	wg.Add(req.Count)
	go a.flushBatchCompletionUpdates(batchID, updates, run.done)

	for itemID := 0; itemID < req.Count; itemID++ {
		go func(itemID int) {
			defer wg.Done()
			a.runBatchCompletionItem(ctx, req, batchID, itemID, updates)
		}(itemID)
	}

	go func() {
		wg.Wait()
		close(updates)
		<-run.done
		a.batchCompletionMu.Lock()
		delete(a.batchCompletionRuns, batchID)
		a.batchCompletionMu.Unlock()
		a.batchCompletionEmit(batchCompletionFinishedEvent, BatchCompletionFinishedEvent{BatchID: batchID})
	}()

	return batchID, nil
}

func (a *App) StopBatchCompletions(batchID string) error {
	a.ensureBatchCompletionState()
	a.batchCompletionMu.Lock()
	run := a.batchCompletionRuns[batchID]
	if run != nil {
		delete(a.batchCompletionRuns, batchID)
	}
	a.batchCompletionMu.Unlock()
	if run == nil {
		return fmt.Errorf("batch %q not found", batchID)
	}
	run.cancel()
	return nil
}

func (req BatchCompletionRequest) validate() error {
	if strings.TrimSpace(req.URL) == "" {
		return errors.New("url is required")
	}
	if req.Count < 1 || req.Count > 1000 {
		return errors.New("count must be between 1 and 1000")
	}
	if req.Body == nil {
		return errors.New("body is required")
	}
	return nil
}

func (a *App) runBatchCompletionItem(ctx context.Context, req BatchCompletionRequest, batchID string, itemID int, updates chan<- BatchCompletionUpdate) {
	updates <- BatchCompletionUpdate{ItemID: itemID, Status: "running"}

	body := cloneCompletionBody(req.Body)
	body["stream"] = true
	encodedBody, err := json.Marshal(body)
	if err != nil {
		updates <- BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error()}
		return
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, req.URL, bytes.NewReader(encodedBody))
	if err != nil {
		updates <- BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error()}
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}

	res, err := batchCompletionHTTPClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			updates <- BatchCompletionUpdate{ItemID: itemID, Status: "aborted"}
			return
		}
		updates <- BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error()}
		return
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
		updates <- BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: strings.TrimSpace(string(message))}
		return
	}

	var text strings.Builder
	scanner := bufio.NewScanner(res.Body)
	buffer := make([]byte, 0, 64*1024)
	scanner.Buffer(buffer, 1024*1024)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			updates <- BatchCompletionUpdate{ItemID: itemID, Status: "aborted", Text: text.String()}
			return
		default:
		}
		delta, done, err := parseCompletionSSELine(scanner.Bytes())
		if err != nil {
			updates <- BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error(), Text: text.String()}
			return
		}
		if done {
			updates <- BatchCompletionUpdate{ItemID: itemID, Status: "done", Text: text.String()}
			return
		}
		if delta != "" {
			text.WriteString(delta)
			updates <- BatchCompletionUpdate{ItemID: itemID, Status: "running", Delta: delta}
		}
	}
	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil {
			updates <- BatchCompletionUpdate{ItemID: itemID, Status: "aborted", Text: text.String()}
			return
		}
		updates <- BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error(), Text: text.String()}
		return
	}
	updates <- BatchCompletionUpdate{ItemID: itemID, Status: "done", Text: text.String()}
}

func (a *App) flushBatchCompletionUpdates(batchID string, updates <-chan BatchCompletionUpdate, done chan<- struct{}) {
	defer close(done)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	buffer := make([]BatchCompletionUpdate, 0, 256)
	flush := func() {
		if len(buffer) == 0 {
			return
		}
		payload := BatchCompletionUpdateEvent{BatchID: batchID, Updates: append([]BatchCompletionUpdate(nil), buffer...)}
		buffer = buffer[:0]
		a.batchCompletionEmit(batchCompletionUpdateEvent, payload)
	}
	for {
		select {
		case update, ok := <-updates:
			if !ok {
				flush()
				return
			}
			buffer = append(buffer, update)
			if len(buffer) >= 256 {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}

func parseCompletionSSELine(line []byte) (string, bool, error) {
	text := strings.TrimSpace(string(line))
	if text == "" || strings.HasPrefix(text, ":") {
		return "", false, nil
	}
	if !strings.HasPrefix(text, "data:") {
		return "", false, nil
	}
	data := strings.TrimSpace(strings.TrimPrefix(text, "data:"))
	if data == "[DONE]" {
		return "", true, nil
	}
	var payload struct {
		Choices []struct {
			Text  string `json:"text"`
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(data), &payload); err != nil {
		return "", false, err
	}
	if len(payload.Choices) == 0 {
		return "", false, nil
	}
	if payload.Choices[0].Text != "" {
		return payload.Choices[0].Text, false, nil
	}
	return payload.Choices[0].Delta.Content, false, nil
}

func cloneCompletionBody(body map[string]any) map[string]any {
	cloned := make(map[string]any, len(body))
	for key, value := range body {
		cloned[key] = value
	}
	return cloned
}

func newBatchCompletionID() string {
	var bytes [8]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(bytes[:])
}
```

- [ ] **Step 2: Add registry fields to `backend-golang/app.go`**

Modify `App`:

```go
type App struct {
	ctx           context.Context
	HasConfigData bool
	ConfigData    map[string]any
	Dev           bool
	proxyPort     int
	exDir         string
	cmdPrefix     string

	batchCompletionMu   sync.Mutex
	batchCompletionRuns map[string]*batchCompletionRun
	batchCompletionEmit batchCompletionEmitFunc
}
```

Add `sync` to the imports in `backend-golang/app.go`.

- [ ] **Step 3: Run Go tests**

Run:

```powershell
go test ./backend-golang -run BatchCompletion
```

Expected: `ok rwkv-runner/backend-golang`.

- [ ] **Step 4: Commit Go backend service**

```powershell
git add backend-golang/app.go backend-golang/batch_completion.go backend-golang/batch_completion_test.go
git commit -m "backend: add client batch completion runner"
```

---

### Task 3: Generate Wails Bindings

**Files:**
- Modify generated: `frontend/wailsjs/go/backend_golang/App.d.ts`
- Modify generated: `frontend/wailsjs/go/backend_golang/App.js`
- Modify generated: `frontend/wailsjs/go/models.ts`

- [ ] **Step 1: Run Wails generate**

Run:

```powershell
wails generate
```

Expected:

- `StartBatchCompletions` appears in `frontend/wailsjs/go/backend_golang/App.d.ts`.
- `StopBatchCompletions` appears in `frontend/wailsjs/go/backend_golang/App.d.ts`.
- `backend_golang.BatchCompletionRequest` appears in `frontend/wailsjs/go/models.ts`.

- [ ] **Step 2: Inspect generated diff**

Run:

```powershell
git diff -- frontend/wailsjs/go/backend_golang/App.d.ts frontend/wailsjs/go/backend_golang/App.js frontend/wailsjs/go/models.ts
```

Expected: generated method and type additions only.

- [ ] **Step 3: Commit generated bindings**

```powershell
git add frontend/wailsjs/go/backend_golang/App.d.ts frontend/wailsjs/go/backend_golang/App.js frontend/wailsjs/go/models.ts
git commit -m "frontend: update batch completion bindings"
```

---

### Task 4: Frontend Batch Types and Client Bridge

**Files:**
- Create: `frontend/src/pages/CompletionBatch/types.ts`
- Create: `frontend/src/pages/CompletionBatch/batchCompletionClient.ts`

- [ ] **Step 1: Add frontend types**

Create `frontend/src/pages/CompletionBatch/types.ts`:

```ts
export type BatchCompletionStatus =
  | 'pending'
  | 'running'
  | 'done'
  | 'error'
  | 'aborted'

export type BatchCompletionItem = {
  id: number
  status: BatchCompletionStatus
  text: string
  error?: string
}

export type BatchCompletionUpdate = {
  itemId: number
  status: BatchCompletionStatus
  delta?: string
  text?: string
  error?: string
}

export type BatchCompletionUpdateEvent = {
  batchId: string
  updates: BatchCompletionUpdate[]
}

export type BatchCompletionFinishedEvent = {
  batchId: string
}
```

- [ ] **Step 2: Add Wails bridge**

Create `frontend/src/pages/CompletionBatch/batchCompletionClient.ts`:

```ts
import {
  StartBatchCompletions,
  StopBatchCompletions,
} from '../../../wailsjs/go/backend_golang/App'
import { EventsOn } from '../../../wailsjs/runtime'
import { backend_golang } from '../../../wailsjs/go/models'
import {
  BatchCompletionFinishedEvent,
  BatchCompletionUpdateEvent,
} from './types'

export const startBatchCompletions = (
  request: backend_golang.BatchCompletionRequest
) => StartBatchCompletions(request)

export const stopBatchCompletions = (batchId: string) =>
  StopBatchCompletions(batchId)

export const subscribeBatchCompletionUpdates = (
  onUpdate: (event: BatchCompletionUpdateEvent) => void,
  onFinished: (event: BatchCompletionFinishedEvent) => void
) => {
  const offUpdate = EventsOn('batch-completion-update', onUpdate)
  const offFinished = EventsOn('batch-completion-finished', onFinished)
  return () => {
    offUpdate()
    offFinished()
  }
}
```

- [ ] **Step 3: Run TypeScript to verify generated imports**

Run:

```powershell
cd frontend
npx tsc --noEmit
```

Expected: if Task 3 bindings are generated correctly, imports resolve. UI files are not using this bridge yet.

- [ ] **Step 4: Commit types and bridge**

```powershell
git add frontend/src/pages/CompletionBatch/types.ts frontend/src/pages/CompletionBatch/batchCompletionClient.ts
git commit -m "frontend: add batch completion client bridge"
```

---

### Task 5: Virtual Grid Helper

**Files:**
- Create: `frontend/src/pages/CompletionBatch/virtualGrid.ts`
- Create: `frontend/src/pages/CompletionBatch/virtualGrid.test.ts` if a frontend test runner is available

- [ ] **Step 1: Add pure virtual range helper**

Create `frontend/src/pages/CompletionBatch/virtualGrid.ts`:

```ts
export type VirtualGridRange = {
  startIndex: number
  endIndex: number
  columns: number
  rowHeight: number
  totalHeight: number
  offsetY: number
}

export const getVirtualGridRange = ({
  itemCount,
  containerWidth,
  viewportHeight,
  scrollTop,
  cellWidth,
  rowHeight,
  overscanRows = 2,
}: {
  itemCount: number
  containerWidth: number
  viewportHeight: number
  scrollTop: number
  cellWidth: number
  rowHeight: number
  overscanRows?: number
}): VirtualGridRange => {
  const columns = Math.max(1, Math.floor(containerWidth / cellWidth))
  const rowCount = Math.ceil(itemCount / columns)
  const firstVisibleRow = Math.max(0, Math.floor(scrollTop / rowHeight) - overscanRows)
  const lastVisibleRow = Math.min(
    rowCount - 1,
    Math.ceil((scrollTop + viewportHeight) / rowHeight) + overscanRows
  )
  const startIndex = Math.min(itemCount, firstVisibleRow * columns)
  const endIndex = Math.min(itemCount, (lastVisibleRow + 1) * columns)
  return {
    startIndex,
    endIndex,
    columns,
    rowHeight,
    totalHeight: rowCount * rowHeight,
    offsetY: firstVisibleRow * rowHeight,
  }
}

export const shouldRenderFullText = (
  itemIndex: number,
  hoveredIndex: number | null
) =>
  hoveredIndex !== null &&
  Math.abs(itemIndex - hoveredIndex) <= 1
```

- [ ] **Step 2: Add helper tests if Vitest or Jest is available**

Create `frontend/src/pages/CompletionBatch/virtualGrid.test.ts`:

```ts
import { getVirtualGridRange, shouldRenderFullText } from './virtualGrid'

test('getVirtualGridRange returns visible overscanned item range', () => {
  const range = getVirtualGridRange({
    itemCount: 1000,
    containerWidth: 500,
    viewportHeight: 200,
    scrollTop: 300,
    cellWidth: 100,
    rowHeight: 50,
    overscanRows: 1,
  })

  expect(range.columns).toBe(5)
  expect(range.startIndex).toBe(25)
  expect(range.endIndex).toBe(60)
  expect(range.totalHeight).toBe(10000)
})

test('shouldRenderFullText only enables hovered and adjacent cells', () => {
  expect(shouldRenderFullText(9, 10)).toBe(true)
  expect(shouldRenderFullText(10, 10)).toBe(true)
  expect(shouldRenderFullText(11, 10)).toBe(true)
  expect(shouldRenderFullText(12, 10)).toBe(false)
  expect(shouldRenderFullText(10, null)).toBe(false)
})
```

If no frontend test runner is configured, skip committing `virtualGrid.test.ts` and rely on `npx tsc --noEmit`; do not add a new test dependency.

- [ ] **Step 3: Run verification**

Run:

```powershell
cd frontend
npx tsc --noEmit
```

Expected: TypeScript passes.

- [ ] **Step 4: Commit virtual grid helper**

```powershell
git add frontend/src/pages/CompletionBatch/virtualGrid.ts
git commit -m "frontend: add batch completion virtual grid helper"
```

---

### Task 6: BatchCompletionOverlay Component

**Files:**
- Create: `frontend/src/pages/CompletionBatch/BatchCompletionOverlay.tsx`
- Modify: `frontend/src/pages/CompletionBatch/types.ts` if reducer actions need exported names

- [ ] **Step 1: Create overlay component**

Create `frontend/src/pages/CompletionBatch/BatchCompletionOverlay.tsx`:

```tsx
import React, {
  FC,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react'
import { Button, Input, Text } from '@fluentui/react-components'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import { backend_golang } from '../../../wailsjs/go/models'
import commonStore from '../../stores/commonStore'
import { CompletionParams, StopItem } from '../../types/completion'
import { getReqUrl } from '../../utils'
import { defaultPenaltyDecay } from '../defaultConfigs'
import {
  startBatchCompletions,
  stopBatchCompletions,
  subscribeBatchCompletionUpdates,
} from './batchCompletionClient'
import { BatchCompletionItem, BatchCompletionUpdateEvent } from './types'
import { getVirtualGridRange, shouldRenderFullText } from './virtualGrid'

type State = {
  batchId: string | null
  items: BatchCompletionItem[]
}

type Action =
  | { type: 'start'; batchId: string; count: number }
  | { type: 'updates'; event: BatchCompletionUpdateEvent }
  | { type: 'finished'; batchId: string }
  | { type: 'clear' }

const createItems = (count: number): BatchCompletionItem[] =>
  Array.from({ length: count }, (_, index) => ({
    id: index,
    status: 'pending',
    text: '',
  }))

const reducer = (state: State, action: Action): State => {
  if (action.type === 'start') {
    return { batchId: action.batchId, items: createItems(action.count) }
  }
  if (action.type === 'clear') {
    return { batchId: null, items: [] }
  }
  if (action.type === 'finished') {
    if (state.batchId !== action.batchId) return state
    return { ...state, batchId: null }
  }
  if (action.type === 'updates') {
    if (state.batchId !== action.event.batchId) return state
    const items = state.items.slice()
    for (const update of action.event.updates) {
      const current = items[update.itemId]
      if (!current) continue
      items[update.itemId] = {
        ...current,
        status: update.status,
        text:
          update.text !== undefined
            ? update.text
            : current.text + (update.delta ?? ''),
        error: update.error,
      }
    }
    return { ...state, items }
  }
  return state
}

export const BatchCompletionOverlay: FC<{
  prompt: string
  params: CompletionParams
  stopItems: StopItem[]
  port: number
}> = ({ prompt, params, stopItems, port }) => {
  const { t } = useTranslation()
  const [count, setCount] = useState(16)
  const [state, dispatch] = useReducer(reducer, { batchId: null, items: [] })
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [size, setSize] = useState({ width: 1, height: 1 })
  const viewportRef = useRef<HTMLDivElement>(null)

  const running = state.items.some(
    (item) => item.status === 'pending' || item.status === 'running'
  )

  useEffect(() => {
    return subscribeBatchCompletionUpdates(
      (event) => dispatch({ type: 'updates', event }),
      (event) => dispatch({ type: 'finished', batchId: event.batchId })
    )
  }, [])

  useEffect(() => {
    const element = viewportRef.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    return () => {
      if (state.batchId) stopBatchCompletions(state.batchId)
    }
  }, [state.batchId])

  const range = useMemo(
    () =>
      getVirtualGridRange({
        itemCount: state.items.length,
        containerWidth: size.width,
        viewportHeight: size.height,
        scrollTop,
        cellWidth: 124,
        rowHeight: 76,
        overscanRows: 3,
      }),
    [state.items.length, scrollTop, size]
  )

  const visibleItems = state.items.slice(range.startIndex, range.endIndex)

  const buildRequest = async (): Promise<backend_golang.BatchCompletionRequest> => {
    const stopSequences = stopItems
      .filter((item) => item.type === 'text')
      .map((item) => item.value.replaceAll('\\n', '\n'))
      .filter((value) => value.length > 0)
    const stopTokenIds = stopItems
      .filter((item) => item.type === 'token')
      .map((item) => Number.parseInt(item.value, 10))
      .filter((value) => Number.isFinite(value) && value >= 0)
    const requestPrompt = prompt + params.injectStart.replaceAll('\\n', '\n')
    const { url, headers } = await getReqUrl(port, '/v1/completions', true)
    return {
      url,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${commonStore.settings.apiKey}`,
        ...headers,
      },
      count,
      body: {
        prompt: requestPrompt,
        stream: true,
        model: commonStore.settings.apiCompletionModelName,
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
      },
    }
  }

  const onClick = async () => {
    if (running && state.batchId) {
      await stopBatchCompletions(state.batchId)
      dispatch({ type: 'finished', batchId: state.batchId })
      return
    }
    try {
      const request = await buildRequest()
      const batchId = await startBatchCompletions(request)
      dispatch({ type: 'start', batchId, count })
    } catch (error: any) {
      toast(error?.message || String(error), { type: 'error' })
    }
  }

  if (commonStore.platform === 'web') return null

  return (
    <div className="pointer-events-none absolute inset-x-3 bottom-3 h-44 rounded border border-neutral-300 bg-white/95 shadow-lg">
      <div className="pointer-events-auto flex h-10 items-center gap-2 border-b border-neutral-200 px-2">
        <Text size={200}>{t('Batch Generation')}</Text>
        <Input
          className="w-20"
          type="number"
          min={1}
          max={1000}
          value={String(count)}
          disabled={running}
          onChange={(_, data) => {
            const parsed = Number.parseInt(data.value, 10)
            setCount(Math.min(1000, Math.max(1, Number.isFinite(parsed) ? parsed : 1)))
          }}
        />
        <Button size="small" appearance={running ? 'secondary' : 'primary'} onClick={onClick}>
          {running ? t('Stop Batch Generation') : t('Batch Generate')}
        </Button>
        <Button size="small" disabled={running} onClick={() => dispatch({ type: 'clear' })}>
          {t('Clear')}
        </Button>
      </div>
      <div
        ref={viewportRef}
        className="pointer-events-auto h-[calc(100%-2.5rem)] overflow-auto px-2 py-2"
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
      >
        <div style={{ height: range.totalHeight, position: 'relative' }}>
          <div
            className="grid gap-2"
            style={{
              gridTemplateColumns: `repeat(${range.columns}, minmax(0, 1fr))`,
              transform: `translateY(${range.offsetY}px)`,
            }}
          >
            {visibleItems.map((item) => {
              const expanded = hoveredIndex === item.id
              const renderText = shouldRenderFullText(item.id, hoveredIndex)
              return (
                <div
                  key={item.id}
                  className={
                    'h-16 rounded border bg-slate-50 p-2 text-xs transition-transform duration-150 ' +
                    (expanded ? 'z-10 scale-150 bg-white shadow-xl' : 'scale-100')
                  }
                  onMouseEnter={() => setHoveredIndex(item.id)}
                  onMouseLeave={() => setHoveredIndex(null)}
                >
                  <div className="flex justify-between text-[11px] text-slate-500">
                    <span>#{item.id + 1}</span>
                    <span>{t(item.status)}</span>
                  </div>
                  {renderText ? (
                    <div className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words text-slate-900 select-text">
                      {item.error || item.text || t('Generating')}
                    </div>
                  ) : (
                    <div className="mt-2 h-3 rounded bg-slate-200" />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Run TypeScript**

Run:

```powershell
cd frontend
npx tsc --noEmit
```

Expected: TypeScript may report long-line formatting only if lint is separate; `tsc` should pass.

- [ ] **Step 3: Commit overlay**

```powershell
git add frontend/src/pages/CompletionBatch/BatchCompletionOverlay.tsx
git commit -m "frontend: add batch completion overlay"
```

---

### Task 7: Integrate Overlay Into Completion Page

**Files:**
- Modify: `frontend/src/pages/Completion.tsx`
- Modify if needed: locale JSON files

- [ ] **Step 1: Import overlay**

Add to `frontend/src/pages/Completion.tsx`:

```ts
import { BatchCompletionOverlay } from './CompletionBatch/BatchCompletionOverlay'
```

- [ ] **Step 2: Wrap the textarea in a relative container**

Replace the current direct textarea block:

```tsx
<Textarea
  ref={inputRef}
  className="grow"
  value={prompt}
  onChange={(e) => {
    commonStore.setCompletionSubmittedPrompt(e.target.value)
    setPrompt(e.target.value)
  }}
/>
```

with:

```tsx
<div className="relative flex min-h-0 grow">
  <Textarea
    ref={inputRef}
    className="grow"
    value={prompt}
    onChange={(e) => {
      commonStore.setCompletionSubmittedPrompt(e.target.value)
      setPrompt(e.target.value)
    }}
  />
  <BatchCompletionOverlay
    prompt={prompt}
    params={params}
    stopItems={stopItems}
    port={port}
  />
</div>
```

- [ ] **Step 3: Add locale strings**

Add to `frontend/src/_locales/zh-hans/main.json`:

```json
"Batch Generation": "批量生成",
"Batch Generate": "批量生成",
"Stop Batch Generation": "停止批量生成",
"Clear": "清空",
"pending": "等待中",
"running": "生成中",
"done": "已完成",
"error": "失败",
"aborted": "已停止",
"Generating": "生成中"
```

Add to `frontend/src/_locales/ja/main.json`:

```json
"Batch Generation": "一括生成",
"Batch Generate": "一括生成",
"Stop Batch Generation": "一括生成を停止",
"Clear": "クリア",
"pending": "待機中",
"running": "生成中",
"done": "完了",
"error": "失敗",
"aborted": "停止済み",
"Generating": "生成中"
```

- [ ] **Step 4: Run formatting and TypeScript**

Run:

```powershell
cd frontend
npx prettier --write src/pages/Completion.tsx src/pages/CompletionBatch src/_locales/zh-hans/main.json src/_locales/ja/main.json
npx tsc --noEmit
```

Expected: Prettier completes; TypeScript passes.

- [ ] **Step 5: Commit integration**

```powershell
git add frontend/src/pages/Completion.tsx frontend/src/pages/CompletionBatch frontend/src/_locales/zh-hans/main.json frontend/src/_locales/ja/main.json
git commit -m "frontend: integrate batch completion demo"
```

---

### Task 8: Desktop Verification

**Files:**
- No source changes expected unless verification finds a bug

- [ ] **Step 1: Run backend and frontend build checks**

Run:

```powershell
go test ./backend-golang -run BatchCompletion
cd frontend
npx tsc --noEmit
```

Expected: both pass.

- [ ] **Step 2: Generate Wails bindings if stale**

Run:

```powershell
wails generate
git diff -- frontend/wailsjs/go/backend_golang/App.d.ts frontend/wailsjs/go/backend_golang/App.js frontend/wailsjs/go/models.ts
```

Expected: no diff after Task 3. If there is a diff, inspect and commit generated files.

- [ ] **Step 3: Manual desktop checks**

Start the desktop app using the repository's normal Wails development command:

```powershell
wails dev
```

Manual checklist:

- Completion page shows batch controls in desktop mode.
- Web mode does not show batch controls.
- Count clamps to 1 and 1000.
- Clicking Batch Generate starts all requested items.
- Clicking Stop Batch Generation aborts unfinished items.
- Completed cells keep their text after stop.
- Main textarea content does not change from batch output.
- Hovering a cell expands it.
- Full text appears only for hovered, previous, and next cells.
- Expanded text can be selected and right-clicked.
- 1000 cells scroll smoothly enough for demo use.

- [ ] **Step 4: Final commit for verification fixes**

If verification required fixes:

```powershell
git add <changed-files>
git commit -m "fix: polish batch completion demo"
```

If verification required no fixes, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Desktop-only behavior is covered by Task 6 and Task 7.
- Go-side fan-out and cancellation are covered by Task 1 and Task 2.
- Wails event communication is covered by Task 2, Task 3, and Task 4.
- Virtualized overlay and max-three text rendering are covered by Task 5 and Task 6.
- Existing Completion textarea isolation is covered by Task 7 and Task 8.
- Verification is covered by Task 8.

Placeholder scan:

- The plan does not rely on unspecified implementation work.
- Every new function used in subsequent tasks is introduced before it is referenced.
- The only conditional path is frontend helper tests, where the plan explicitly avoids adding new dependencies if no runner exists.

Type consistency:

- Go method names match Wails imports: `StartBatchCompletions`, `StopBatchCompletions`.
- Event names match frontend subscriptions: `batch-completion-update`, `batch-completion-finished`.
- Status strings match frontend union type: `pending`, `running`, `done`, `error`, `aborted`.
