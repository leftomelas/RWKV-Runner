package backend_golang

import (
	"bufio"
	"encoding/json"
	"io"
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
	updateEvents := make(chan BatchCompletionUpdateEvent, 8)
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
		payload, ok := data.(BatchCompletionUpdateEvent)
		if !ok {
			return
		}
		updateEvents <- payload
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

	select {
	case payload := <-updateEvents:
		if payload.BatchID != batchID {
			t.Fatalf("update batch id = %q, want %q", payload.BatchID, batchID)
		}
		if len(payload.Updates) == 0 {
			t.Fatal("expected at least one update")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected at least one batch completion update event")
	}
}

func TestBatchCompletionStopCancelsActiveRun(t *testing.T) {
	started := make(chan struct{}, 1)
	canceled := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		started <- struct{}{}
		select {
		case <-r.Context().Done():
			canceled <- struct{}{}
		case <-time.After(5 * time.Second):
		}
	}))
	defer server.Close()

	app := NewApp()
	batchID, err := app.StartBatchCompletions(BatchCompletionRequest{
		URL:   server.URL,
		Count: 1,
		Body:  map[string]any{"prompt": "hello", "stream": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("request did not start")
	}

	if err := app.StopBatchCompletions(batchID); err != nil {
		t.Fatal(err)
	}
	select {
	case <-canceled:
	case <-time.After(2 * time.Second):
		t.Fatal("active request was not canceled")
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
