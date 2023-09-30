FROM python:3.11-slim

WORKDIR /app
ENV MLSERVER_PATH=/app

RUN useradd -u 1000 -s /bin/bash mlserver -d $MLSERVER_PATH && \
    chown -R 1000:0 $MLSERVER_PATH && \
    chmod -R 776 $MLSERVER_PATH

USER mlserver

COPY --chown=mlserver:mlserver model-settings.json .
COPY --chown=mlserver:mlserver requirements.txt .

RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

RUN . venv/bin/activate && \
    pip install mlserver-mlflow==1.3.5

ENV PATH=/app/venv/bin:$PATH

# Inference
EXPOSE 8080
# gRPC
EXPOSE 8081
# Prometheus
EXPOSE 8082

# Use the shell script as the ENTRYPOINT
ENTRYPOINT ["mlserver"]
CMD [ "start", "." ]

