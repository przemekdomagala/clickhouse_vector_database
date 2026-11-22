FROM clickhouse/clickhouse-server:26.10.9.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY config/clickhouse.xml /etc/clickhouse-server/config.d/benchmark.xml
COPY sql /app/sql
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.py && ln -s /app/scripts/db.py /usr/local/bin/db \
    && mkdir -p /srv/user_files && ln -s /data /srv/user_files/data && chown -R clickhouse:clickhouse /srv/user_files

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/scripts \
    CLICKHOUSE_SKIP_USER_SETUP=1
