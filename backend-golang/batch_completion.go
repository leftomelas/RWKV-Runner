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

var batchCompletionHTTPClient = &http.Client{
	Timeout: 0,
	Transport: &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		MaxIdleConns:        1200,
		MaxIdleConnsPerHost: 1200,
		IdleConnTimeout:     90 * time.Second,
	},
}

var batchCompletionRetryDelays = []time.Duration{
	50 * time.Millisecond,
	100 * time.Millisecond,
	200 * time.Millisecond,
	400 * time.Millisecond,
	800 * time.Millisecond,
	1200 * time.Millisecond,
	1800 * time.Millisecond,
	2500 * time.Millisecond,
	3500 * time.Millisecond,
	5000 * time.Millisecond,
}

func (a *App) StartBatchCompletions(req BatchCompletionRequest) (string, error) {
	if err := req.validate(); err != nil {
		return "", err
	}

	batchID := newBatchCompletionID()
	parentCtx := a.ctx
	if parentCtx == nil {
		parentCtx = context.Background()
	}
	ctx, cancel := context.WithCancel(parentCtx)
	run := &batchCompletionRun{
		cancel: cancel,
		done:   make(chan struct{}),
	}

	a.batchCompletionMu.Lock()
	a.ensureBatchCompletionStateLocked()
	a.batchCompletionRuns[batchID] = run
	a.batchCompletionMu.Unlock()

	updates := make(chan BatchCompletionUpdate, 4096)
	var wg sync.WaitGroup
	wg.Add(req.Count)

	go a.flushBatchCompletionUpdates(batchID, updates, run.done)

	for itemID := 0; itemID < req.Count; itemID++ {
		go func(itemID int) {
			defer wg.Done()
			a.runBatchCompletionItem(ctx, req, itemID, updates)
		}(itemID)
	}

	go func() {
		wg.Wait()
		close(updates)
		<-run.done

		a.batchCompletionMu.Lock()
		delete(a.batchCompletionRuns, batchID)
		emit := a.batchCompletionEmit
		a.batchCompletionMu.Unlock()

		emit(batchCompletionFinishedEvent, BatchCompletionFinishedEvent{BatchID: batchID})
	}()

	return batchID, nil
}

func (a *App) StopBatchCompletions(batchID string) error {
	a.batchCompletionMu.Lock()
	a.ensureBatchCompletionStateLocked()
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

func (a *App) stopAllBatchCompletions() {
	a.batchCompletionMu.Lock()
	for batchID, run := range a.batchCompletionRuns {
		delete(a.batchCompletionRuns, batchID)
		run.cancel()
	}
	a.batchCompletionMu.Unlock()
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

func (a *App) ensureBatchCompletionStateLocked() {
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

func (a *App) runBatchCompletionItem(ctx context.Context, req BatchCompletionRequest, itemID int, updates chan<- BatchCompletionUpdate) {
	sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "running"})

	body := cloneCompletionBody(req.Body)
	body["stream"] = true

	encodedBody, err := json.Marshal(body)
	if err != nil {
		sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error()})
		return
	}

	res, err := doBatchCompletionRequest(ctx, req, itemID, encodedBody)
	if err != nil {
		if ctx.Err() != nil {
			sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "aborted"})
			return
		}
		sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error()})
		return
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
		errorMessage := strings.TrimSpace(string(message))
		if errorMessage == "" {
			errorMessage = res.Status
		}
		sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: errorMessage})
		return
	}

	var text strings.Builder
	scanner := bufio.NewScanner(res.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "aborted", Text: text.String()})
			return
		default:
		}

		delta, done, err := parseCompletionSSELine(scanner.Bytes())
		if err != nil {
			sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error(), Text: text.String()})
			return
		}
		if done {
			sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "done", Text: text.String()})
			return
		}
		if delta == "" {
			continue
		}

		text.WriteString(delta)
		sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "running", Delta: delta})
	}

	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil {
			sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "aborted", Text: text.String()})
			return
		}
		sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "error", Error: err.Error(), Text: text.String()})
		return
	}

	sendBatchCompletionUpdate(ctx, updates, BatchCompletionUpdate{ItemID: itemID, Status: "done", Text: text.String()})
}

func doBatchCompletionRequest(ctx context.Context, req BatchCompletionRequest, itemID int, encodedBody []byte) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= len(batchCompletionRetryDelays); attempt++ {
		if attempt == 0 {
			if err := waitBatchCompletionRetryDelay(ctx, initialBatchCompletionConnectDelay(itemID)); err != nil {
				return nil, err
			}
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, req.URL, bytes.NewReader(encodedBody))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		for key, value := range req.Headers {
			httpReq.Header.Set(key, value)
		}

		res, err := batchCompletionHTTPClient.Do(httpReq)
		if err == nil {
			return res, nil
		}
		lastErr = err
		if ctx.Err() != nil || !isRetriableBatchCompletionDialError(err) || attempt == len(batchCompletionRetryDelays) {
			return nil, err
		}

		delay := batchCompletionRetryDelays[attempt] + batchCompletionRetryJitter(itemID, attempt)
		if err := waitBatchCompletionRetryDelay(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func initialBatchCompletionConnectDelay(itemID int) time.Duration {
	return time.Duration(itemID%128) * 4 * time.Millisecond
}

func batchCompletionRetryJitter(itemID int, attempt int) time.Duration {
	return time.Duration((itemID*17+attempt*31)%97) * 11 * time.Millisecond
}

func waitBatchCompletionRetryDelay(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		timer.Stop()
		return ctx.Err()
	}
}

func isRetriableBatchCompletionDialError(err error) bool {
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "connection refused") ||
		strings.Contains(message, "actively refused") ||
		strings.Contains(message, "connection reset by peer") ||
		strings.Contains(message, "forcibly closed")
}

func sendBatchCompletionUpdate(ctx context.Context, updates chan<- BatchCompletionUpdate, update BatchCompletionUpdate) bool {
	select {
	case updates <- update:
		return true
	case <-ctx.Done():
		return false
	}
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

		payload := BatchCompletionUpdateEvent{
			BatchID: batchID,
			Updates: append([]BatchCompletionUpdate(nil), buffer...),
		}
		buffer = buffer[:0]

		a.batchCompletionMu.Lock()
		emit := a.batchCompletionEmit
		a.batchCompletionMu.Unlock()
		emit(batchCompletionUpdateEvent, payload)
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
	if text == "" || strings.HasPrefix(text, ":") || !strings.HasPrefix(text, "data:") {
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
