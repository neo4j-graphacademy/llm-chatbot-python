# SPDX-FileCopyrightText: 2024 Helmholtz Centre for Environmental Research (UFZ)
#
# SPDX-License-Identifier: AGPL-3.0-only

FROM python:3.12 as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && apt upgrade -y -qq

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


FROM python:3.12

RUN apt update && apt upgrade -y -qq \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r ecotoxfred && useradd -r -g ecotoxfred -m -d /home/ecotoxfred -s /bin/sh -c "ecotoxfred User" ecotoxfred
WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache /wheels/*

RUN mkdir -p /app/.config/matplotlib && chown -R ecotoxfred:ecotoxfred /app
ENV MPLCONFIGDIR /app/.config/matplotlib

# Change User to a Non-Root user
USER ecotoxfred

# Copy your source code to image workdir
COPY *.py .
COPY figures/ figures/
COPY tools/ tools/

# indicates the container port, which is exposed to the host
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# The command which should be used when container starts
ENTRYPOINT ["streamlit", "run", "bot.py", "--server.port=8501", "--server.address=0.0.0.0"]
