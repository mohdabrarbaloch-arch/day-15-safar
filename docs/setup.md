# Safar - setup

Two ways to run it: Docker (nothing to install) or a local Python environment.

## Docker

```bash
git clone https://github.com/mohdabrarbaloch-arch/day-15-safar.git
cd day-15-safar
cp .env.example .env
docker compose up --build
```

Open http://localhost:8000. The demo account is `demo@safar.pk` with password
`safar-demo-2026`.

The compose file starts Postgres 16 and the app, and waits for the database
health check before booting the app. Data lives in the `safar_pgdata` volume, so
`docker compose down` keeps your runs and `docker compose down -v` wipes them.

## Local Python

```bash
git clone https://github.com/mohdabrarbaloch-arch/day-15-safar.git
cd day-15-safar
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

With no `DATABASE_URL` set, the app uses SQLite and creates `safar.db` on first
boot. No database server needed.

## Environment variables

Every variable is documented in `.env.example`. The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` refuses to boot with a weak `JWT_SECRET` or a wildcard CORS origin |
| `JWT_SECRET` | dev placeholder | Must be at least 32 characters in production |
| `DATABASE_URL` | `sqlite:///./safar.db` | Postgres URL for production |
| `CORS_ORIGINS` | localhost | Comma-separated allow-list |
| `SEED_DEMO` | `true` | Ignored when `ENVIRONMENT=production` |
| `SIM_DT` | `0.05` | Integration step in seconds |
| `SIM_MAX_DURATION` | `120` | Hard ceiling on simulated seconds per run |
| `SIM_MAX_FRAMES` | `4000` | Hard ceiling on frames returned |

Generate a real secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Running the tests

```bash
pytest -q
ruff check .
black --check .
```

The suite runs in a couple of seconds because the simulation core has no I/O.

## Troubleshooting

**`ModuleNotFoundError: No module named 'app'`**
Run from the repository root, and make sure you installed with `pip install -e .`
so the package is on the path.

**`passlib` warns about `bcrypt.__about__`**
Harmless. `pyproject.toml` pins `bcrypt<4.1` to keep the boot clean.

**Postgres connection refused in Docker**
The app waits on the health check, but on a slow first pull the database can
still be initialising. `docker compose up` again once Postgres is up.

**Port 8000 already in use**
`PORT=8001 uvicorn app.main:app` or change the mapping in `docker-compose.yml`.

## Deploying

`vercel.json` is included and targets the Python runtime. Deployment needs a
`VERCEL_TOKEN`:

```bash
npm i -g vercel
vercel --prod --yes --token "$VERCEL_TOKEN"
```

Set `JWT_SECRET`, `DATABASE_URL` and `CORS_ORIGINS` in the Vercel project
settings before the first production deploy. `ENVIRONMENT=production` is already
set in `vercel.json`.
