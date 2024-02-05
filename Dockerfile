# docker build -t aironsuit-linux --build-arg PYPI_USERNAME=<user_name> --build-arg PYPI_PASSWORD=<password> .
FROM python:3.9.18-slim-bullseye as base
ENV TZ="UTC"
RUN apt update && \
    apt install --no-install-recommends -y curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

FROM base as build_base
# Install dependencies.
ENV PYTHONUNBUFFERED True
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get update  \
    && apt-get install -y build-essential \
    && apt-get -y install gcc mono-mcs \
    && rm -rf /var/lib/apt/lists/*

FROM build_base as build_aironsuit
# Copy the code to the container image
ARG PYPI_USERNAME="user_name"
ARG PYPI_PASSWORD="password"
WORKDIR /app
COPY . ./
# Install packages, build the wheel and publish it
RUN poetry install  \
    && poetry export --without-hashes --format=requirements.txt > requirements.txt  \
    && poetry run python setup.py bdist_wheel \
    && poetry run python -m twine upload dist/* --username $PYPI_USERNAME --password $PYPI_PASSWORD