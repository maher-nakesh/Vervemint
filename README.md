# Vervemint - AI that mint-answers from your docs


Ask about your technical documents and get answers with page-level citations — in a web app or on Telegram. 

## What it does

- **Cited answers.** Every claim ends with a source you can open: the exact passage, highlighted inside its page.
- **Honest refusals.** Weak evidence means no answer, not a plausible one.
- **Log diagnosis.** Attach a `.log`, `.csv` or `.txt` from a machine and get a diagnosis based on the manuals.
- **Maintenance agent.** Checks machine telemetry, searches the manuals, and drafts a work order — created only after you approve it.
- **Your own documents.** Upload PDF, TXT or MD files and ask about those instead of the built-in library.
- **Any model.** Ollama (local, no key) or Gemini, Claude, OpenAI. Switch in the sidebar at any time.
- **Telegram.** The same assistant in a chat, for people who never open the web app.

## Quick start

```bash
git clone https://github.com/maher-nakesh/Vervemint.git vervemint
cd vervemint

python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

# The document set (Panasonic manuals), then the search index
python -c "from huggingface_hub import snapshot_download; snapshot_download('Parssky/industrial-instruction-dataset', repo_type='dataset', local_dir='data/industrial-instruction-dataset')"
python -m src.vervemint.index

# A local model, or skip this and use Gemini / Claude / OpenAI instead
ollama pull qwen2.5:7b
```

Start it in two terminals:

```bash
python -m uvicorn src.vervemint.api:app --host 127.0.0.1 --port 8000   # backend
python -m streamlit run ui/app.py                                      # web app
```




## Telegram bot

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Paste it in **Settings → Telegram bot**. The badge turns *Connected @yourbot* within seconds — the backend runs the bot, so there is nothing else to start.
3. Message your bot. It replies with your Telegram user id; add that id under **Allowed users**.

In the chat: a question gets a cited answer, a `.log` / `.csv` / `.txt` file gets a diagnosis, a `.pdf` / `.md` file is added to *My documents*, `/settings` switches mode and documents, and drafted work orders come with **Approve** / **Reject** buttons.

Only allowed users can use it, in private chats only. Clearing the token disconnects it.

## Run with Docker

```bash
docker compose up -d --build
```



## Settings, configuration and secrets

| Where | What | Notes |
| --- | --- | --- |
| **Settings page** | API keys, Ollama address, Telegram bot, web UI password, API token | Saved in `data/credentials.json` (owner-only, gitignored). Applies at once, no restart. |
| `config.yaml` | Models, chunking, retrieval `top_k`, abstain threshold, log level | Edit and restart. Invalid values stop the app with a clear message. |
| Environment | `VERVEMINT_<KEY>` for config; `GEMINI_API_KEY`, `VERVEMINT_TELEGRAM_BOT_TOKEN`, … for secrets | Wins over both, and locks that field in Settings — for secrets managers. |

Only the API token and the web UI password are generated for you, on the first start. Nothing else is written until you set it, so the file always shows your own choices.

## Use the API

```bash
TOKEN=...   # Settings → Access → API token

curl -X POST http://127.0.0.1:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What communication protocol do digital air pressure sensors use?"}'
```

Interactive docs: 
<http://127.0.0.1:8000/docs>. Endpoints: `GET /health` (public), `POST /ask`, `POST /agent`, `POST /analyze-log`, `POST /connect`, `GET|POST /documents`, `DELETE /documents/{id}`, `GET|PUT /settings`, `GET /telegram/status`, `POST /work-orders`, `GET /stats`. Everything except `/health` needs the token. Leave out `provider` / `model` to use the ones chosen in the sidebar.


## Evaluation

**Retrieval** — 914 benchmark questions; a chunk counts as correct if it comes from the gold page.

| Method | Hit@1 | Hit@5 | MRR@10 |
| --- | --- | --- | --- |
| BM25 | 0.365 | 0.570 | 0.458 |
| Dense (bge-small) | **0.485** | 0.770 | 0.600 |
| Hybrid (BM25 + dense, RRF) | 0.384 | 0.717 | 0.540 |
| Hybrid + rerank (used by the app) | 0.447 | **0.837** | **0.605** |


```bash
python -m evaluation.eval_retrieval        # retrieval metrics, no LLM calls
python -m evaluation.eval_answers --n 50   # answer accuracy
python -m pytest tests                     # 76 tests, no LLM calls
```

## Logs

| File | Contents |
| --- | --- |
| `logs/vervemint.log` | One line per request and per LLM call (latency, tokens, Ollama speed), with a request id |
| `logs/llm.log` | The full prompt and reply of every LLM call (`log_llm_messages: false` turns it off) |
| `logs/traces.jsonl` | One record per question; `GET /stats` summarizes abstention rate and latency |

API keys and bot tokens never reach the logs. Questions and document text do, so keep `logs/` private.

## Limits

- **Local 7B answers are weaker** than frontier models, and log analysis takes 25–50 s. Citations prove a source exists, not that it supports every sentence.
- **Retrieval needs a GPU to feel fast**: reranking takes ~0.2 s on a GPU versus seconds on a CPU.
- **Single instance, single tenant**: one shared document store and one set of settings; files on one volume, models in one process.
- **Secrets are stored as readable JSON** on the server so the app can use them — protect the data volume, or inject them from a secrets manager.
- **Guardrails are pattern-based**: they catch obvious prompt injection, not paraphrased attempts.


## Models

| Component | Model |
| --- | --- |
| Dense retrieval | `BAAI/bge-small-en-v1.5` |
| Reranking | `BAAI/bge-reranker-base` |
| Keyword search | `bm25s` |
| Answers (pick one) | Ollama `qwen2.5` 7B · `claude-opus-5` · `gpt-5-mini` · `gemini-3.8-flash` |

## License

Educational / research use.
