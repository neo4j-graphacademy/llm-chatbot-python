# SPDX-FileCopyrightText: 2024 Helmholtz Centre for Environmental Research (UFZ)
#
# SPDX-License-Identifier: AGPL-3.0-only

FROM python:3.10 as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && apt upgrade -y -qq

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


FROM python:3.10

RUN apt update && apt upgrade -y -qq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache /wheels/*

# Change User to a Non-Root user
USER 1000

# Copy your source code to image workdir
COPY *.py .
COPY figures/ figures/
COPY tools/ tools/

# Environment variable to set when container is started (-e MY_OTHER_ENV_VAR=value)
ENV OPENAI_API_KEY ""
ENV OPENAI_MODEL ""
ENV NEO4J_URI ""
ENV NEO4J_USERNAME ""
ENV NEO4J_PASSWORD ""


VOLUME .streamlit
VOLUME /.config/matplotlib

# indicates the container port, which is exposed to the host
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# The command which should be used when container starts
ENTRYPOINT ["streamlit", "run", "bot.py", "--server.port=8501", "--server.address=0.0.0.0"]
