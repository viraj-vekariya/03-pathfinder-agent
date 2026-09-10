# Pathfinder API image.
#
# The CPU-only torch wheel: the default bundles CUDA and is ~2.5GB against ~200MB here.
# Nothing in this service will ever see a GPU - flan-t5-base and MiniLM both run on CPU
# in well under a second.
FROM python:3.12-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir --prefix=/install \
        fastapi "uvicorn[standard]" numpy transformers "psycopg[binary]"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 pathfinder
COPY --from=build /install /usr/local
WORKDIR /app

COPY graph/ ./graph/
COPY rag/ ./rag/
COPY agent/ ./agent/
COPY api/ ./api/
# The graph is BAKED IN rather than fetched at boot. A container that has to reach
# PyPI before it can serve has made a third-party API a hard startup dependency, and
# the numbers in the README describe this snapshot of the graph, not a later one.
COPY data/dependency_graph.json ./data/

RUN mkdir -p /app/outputs && chown -R pathfinder:pathfinder /app
USER pathfinder

ENV PYTHONUNBUFFERED=1 \
    PATHFINDER_GRAPH=/app/data/dependency_graph.json \
    HF_HOME=/app/.cache

EXPOSE 8300
# Long start period: the first boot downloads MiniLM and flan-t5-base. Readiness must
# not kill the container while that is happening.
HEALTHCHECK --interval=20s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8300/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8300"]
