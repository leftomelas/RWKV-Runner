# Completion Batch Generation Design

## Goal

Add a client-only batch generation demo to the Completion page. The feature sends up to 1000 concurrent completion requests through the Go/Wails backend, then renders the results in an animated virtual overlay at the bottom of the existing Completion input area without changing the main input text.

## User Experience

The Completion page keeps its existing textarea, presets, parameters, and normal Generate/Stop flow. A new batch control lets the user enter a count from 1 to 1000 and start batch generation. If any batch item is still pending or running, the same button changes to "Stop Batch Generation" and aborts all unfinished requests.

Batch results appear in an overlay anchored to the bottom of the Completion textarea area. The overlay is independently scrollable with the mouse wheel. Results are presented as a grid of compact cells. Cells normally show lightweight status content only: pending, generating, done, stopped, or error. The currently hovered cell expands, adjacent cells animate around it, and at most three cells render full text content at a time: hovered, previous, and next. Text in expanded cells is read-only DOM text with normal selection and right-click behavior.

The feature is hidden in web mode. It is only available in the desktop client where Wails and the Go backend are present.

## Architecture

The frontend should not create hundreds or thousands of streaming fetch requests. Instead, it calls a Wails method to start a batch and listens for Wails runtime events. The Go backend owns request fan-out, cancellation, SSE parsing, and event aggregation. The frontend owns state presentation, virtual grid layout, hover animation, and text rendering.

This split avoids WebView connection pool limits and reduces per-token JavaScript event pressure. Go emits batched updates at a fixed cadence so the frontend receives fewer, larger update packets.

## Go Backend

Create `backend-golang/batch_completion.go`.

Expose two Wails methods on `App`:

```go
func (a *App) StartBatchCompletions(req BatchCompletionRequest) (string, error)
func (a *App) StopBatchCompletions(batchID string) error
```

`BatchCompletionRequest` includes:

```go
type BatchCompletionRequest struct {
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Count   int               `json:"count"`
	Body    map[string]any    `json:"body"`
}
```

The Go backend validates:

- `Count` must be between 1 and 1000.
- `URL` must be non-empty.
- `Body` must be non-nil.

For every item, Go starts a goroutine immediately. Each goroutine sends one streaming `/v1/completions` request with the same body. It parses SSE `data:` lines, extracts text from `choices[0].text` or `choices[0].delta.content`, and emits status updates through an aggregator channel.

Go keeps an active batch registry:

```go
type batchCompletionRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}
```

`StopBatchCompletions` cancels the batch context. Canceled items emit `aborted` unless already done or failed.

Events:

```go
const batchCompletionUpdateEvent = "batch-completion-update"
const batchCompletionFinishedEvent = "batch-completion-finished"
```

Update payload:

```go
type BatchCompletionUpdateEvent struct {
	BatchID string                  `json:"batchId"`
	Updates []BatchCompletionUpdate `json:"updates"`
}

type BatchCompletionUpdate struct {
	ItemID int    `json:"itemId"`
	Status string `json:"status"`
	Delta  string `json:"delta,omitempty"`
	Text   string `json:"text,omitempty"`
	Error  string `json:"error,omitempty"`
}
```

The aggregator flushes updates every 50 ms or after a moderate buffer threshold such as 256 updates, whichever comes first.

## Frontend

Create a focused component set under `frontend/src/pages/CompletionBatch/`:

- `types.ts`: frontend status and event types.
- `batchCompletionClient.ts`: Wails method calls and event subscription helpers.
- `BatchCompletionOverlay.tsx`: virtual overlay, controls, status cells, hover behavior, and stop/start logic.

Modify `frontend/src/pages/Completion.tsx` only enough to:

- Wrap the textarea in a relative container.
- Render `BatchCompletionOverlay` over the bottom of that container.
- Pass the current prompt, params, stop items, API port, and model name.
- Hide the overlay completely when `commonStore.platform === "web"`.

The overlay should keep its own state with `useReducer` so 1000 result updates remain predictable. It should not write generated text back to `commonStore.completionPreset.prompt`.

Virtualization uses fixed cell measurements and scroll position. The overlay computes the visible row range and renders only visible cells plus overscan. The expanded hovered cell uses CSS transforms and z-index, while full text rendering is limited by:

```ts
const textRenderWindow = new Set([hoveredIndex - 1, hoveredIndex, hoveredIndex + 1])
```

Cells outside that set display status and a compact visual marker only.

## Error Handling

Go item-level errors become `error` updates with a short message. Batch-level validation errors reject `StartBatchCompletions` and should show a toast in the frontend. User stop emits `aborted` for unfinished cells and leaves completed cells unchanged.

Frontend ignores events whose `batchId` does not match the currently active batch. On unmount, the overlay stops any active batch and unsubscribes from events.

## Testing

Go tests cover validation, SSE parsing, event aggregation, and cancellation using `httptest.Server`. Frontend tests should focus on pure virtual range helpers and reducer state transitions where practical. TypeScript and Go builds are required before implementation is considered complete.

Manual verification:

- Desktop mode shows the controls; web mode hides them.
- Count accepts 1 to 1000.
- Starting 1000 items creates 1000 cells without rendering 1000 full text blocks.
- Stop aborts all unfinished items.
- Hovered cell expands and text can be selected.
- Main textarea content remains unchanged after batch generation.
