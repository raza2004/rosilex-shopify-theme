# syntax=docker/dockerfile:1
# Rosilex makeup demo API. Works on Hugging Face Spaces (port 7860) and Render (sets $PORT).

# Build wheels, including insightface's compiled extension, in a throwaway stage.
FROM python:3.11-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY requirements.txt requirements-dev.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-dev.txt

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    INSIGHTFACE_ROOT=/app/models \
    HOME=/home/user
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt
# Bake the buffalo_l face models into the image so startup needs no download.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='/app/models', providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])" \
    && rm -f /app/models/models/buffalo_l.zip
# Offline city and country lookup for the visitor log, so visitor IPs are never sent to a lookup service.
# DB-IP IP to City Lite, CC BY 4.0, https://db-ip.com. Falls back to last month's file early in a month.
RUN <<EOF python
import datetime, gzip, os, urllib.request
first = datetime.date.today().replace(day=1)
months = [first, (first - datetime.timedelta(days=1)).replace(day=1)]
os.makedirs("/app/geo", exist_ok=True)
for month in months:
    url = f"https://download.db-ip.com/free/dbip-city-lite-{month:%Y-%m}.mmdb.gz"
    try:
        # db-ip.com answers 403 to Python's default user agent.
        request = urllib.request.Request(url, headers={"User-Agent": "rosilex-makeup-build/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = gzip.decompress(response.read())
    except Exception as error:
        print(f"{url}: {error}")
        continue
    open("/app/geo/dbip-city-lite.mmdb", "wb").write(data)
    print(f"downloaded {url}")
    break
else:
    raise SystemExit("could not download the DB-IP City Lite database")
EOF

# Tests and evaluation. Mount the service folder at /work:
#   docker build --target dev -t makeup-dev .
#   docker run --rm -v "$PWD:/work" -w /work makeup-dev python -m pytest -q
FROM runtime AS dev
COPY requirements-dev.txt .
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels -r requirements-dev.txt

FROM runtime AS final
COPY app ./app
COPY config ./config
# /data holds the visitor log (VISITORS_FILE=/data/visitors.json). On a free Hugging Face Space this
# disk is temporary: the log starts again when the Space restarts or rebuilds.
RUN mkdir -p /data && chown -R user:user /app /data
USER user
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-7860} --no-access-log"]
