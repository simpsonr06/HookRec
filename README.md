# HookRec — Music Artist Recommendation Engine

HookRec is a reproducible music-artist discovery MVP using the GroupLens HetRec 2011 Last.fm 2K dataset. It combines implicit-feedback item-item collaborative filtering, TF-IDF tag-based content similarity, and a hybrid ranker, with offline ranking evaluation and a FastAPI service.

> **Scope:** artist recommendations, not track-level recommendations. The source dataset contains user–artist listening counts and artist tags.

## Features
- Automated Last.fm 2K dataset download (raw data is not committed)
- Item-item collaborative filtering using log-transformed listening counts
- Content-based artist similarity from aggregated tags
- Hybrid recommendation strategy and popularity fallback for unknown users
- Random leave-one-out evaluation with Precision@K, Recall@K and NDCG@K
- Popularity baseline for comparison
- FastAPI endpoints for recommendations and similar artists
- Unit tests using synthetic fixtures

## Quick start (Python 3.10+)

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_dataset.py
python -m hookrec.evaluate --data-dir data/raw/lastfm-2k --k 10
uvicorn hookrec.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## API
- `GET /health`
- `GET /recommend/{user_id}?k=10&strategy=hybrid`
- `GET /similar/{artist_id}?k=10`

Strategies: `hybrid`, `collaborative`, `content`. Unknown users receive popularity-based recommendations. Artist IDs not present in the dataset return 404 from `/similar`.

## Evaluation protocol
For every user with at least two unique artist interactions, one artist is randomly held out using a fixed seed; remaining interactions form the training set. This is a **random leave-one-out split, not a temporal split**. The evaluator compares hybrid, collaborative, content and popularity strategies. No metric values are claimed until the evaluation is run locally; generated results are saved to `reports/evaluation.json`.

## Limitations
- Historical dataset released in 2011; small and non-commercial-use only.
- Aggregated listening counts, not event-level streaming sessions.
- Random holdout does not simulate chronological recommendation.
- Community tags are sparse and user-generated.
- No neural model, online learning, real-time feedback or track-level recommendation in this MVP.
- Offline ranking metrics do not prove production impact or user satisfaction.

## Tests
```bash
pytest -q
```

## Dataset and attribution
See [DATASET.md](DATASET.md). Official source: https://grouplens.org/datasets/hetrec-2011/ . Cite Last.fm and the HetRec 2011 workshop when using the dataset.

## License
HookRec source code: MIT. Dataset: separate terms; non-commercial use only. The repository's MIT license does not cover third-party data.
