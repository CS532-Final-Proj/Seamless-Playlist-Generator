# Stage 1: Build the virtual environment and application
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV VIRTUAL_ENV=/opt/spark-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Install dependencies
# We use --mount=type=cache to cache uv downloads
COPY pyproject.toml README.md ./

# Create venv
RUN uv venv $VIRTUAL_ENV

# Install dependencies first (cached if pyproject.toml doesn't change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml

# Copy source code
COPY train.py common.py inference.py ./

# Build the project wheel
RUN uv build

# Install the wheel into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install dist/*.whl


# Stage 2: Final image
FROM apache/spark:4.0.1-java21-python3

USER root

# Install ffmpeg for Essentia to handle audio files robustly
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define spark_uid variable (default in apache/spark images is 185)
ARG spark_uid=185

# Copy the virtual environment from the builder stage
ENV VIRTUAL_ENV=/opt/spark-venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Fix venv python symlink because builder has python at /usr/local/bin/python3
# and this image has it at /usr/bin/python3
RUN ln -sf /usr/bin/python3 $VIRTUAL_ENV/bin/python && \
    ln -sf /usr/bin/python3 $VIRTUAL_ENV/bin/python3

# Detect Python version in virtual environment and set PYTHONPATH
RUN PYTHON_VERSION=$(ls $VIRTUAL_ENV/lib | grep python) && \
    echo "export PYTHONPATH=\"$VIRTUAL_ENV/lib/${PYTHON_VERSION}/site-packages:\$PYTHONPATH\"" > /etc/profile.d/spark-venv.sh

# Symlink the installed scripts to the expected work directory
RUN PYTHON_VERSION=$(ls $VIRTUAL_ENV/lib | grep python) && \
    ln -s $VIRTUAL_ENV/lib/${PYTHON_VERSION}/site-packages/*.py /opt/spark/work-dir/

# Download PostgreSQL JDBC driver
ADD https://jdbc.postgresql.org/download/postgresql-42.7.3.jar /opt/spark/jars/
RUN chmod 644 /opt/spark/jars/postgresql-42.7.3.jar

# Download Hadoop AWS and AWS SDK Bundle for S3 support
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar /opt/spark/jars/
# Hadoop 3.4.x introduces support for AWS SDK V2. We include both V1 and V2 bundles to ensure compatibility.
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.780/aws-java-sdk-bundle-1.12.780.jar /opt/spark/jars/
ADD https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.29.23/bundle-2.29.23.jar /opt/spark/jars/
RUN chmod 644 /opt/spark/jars/hadoop-aws-3.4.1.jar /opt/spark/jars/aws-java-sdk-bundle-1.12.780.jar /opt/spark/jars/bundle-2.29.23.jar

# Set proper permissions
RUN chown -R ${spark_uid}:${spark_uid} /opt/spark/work-dir /opt/spark-venv

ENV SPARK_HOME=${SPARK_HOME:-"/opt/spark"}

RUN mkdir -p ${SPARK_HOME} && mkdir -p ${SPARK_HOME}/spark-events
WORKDIR ${SPARK_HOME}

# Spark environment
#
ENV SPARK_MASTER_PORT=7077
ENV SPARK_MASTER_HOST=spark-master
ENV SPARK_MASTER="spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT"

# Add spark binaries to shell and enable execution
RUN chmod u+x /opt/spark/sbin/* && \
    chmod u+x /opt/spark/bin/*
ENV PATH="$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin"

COPY conf/spark-defaults.conf "$SPARK_HOME/conf/"

COPY entrypoint.sh .
RUN chmod u+x /opt/spark/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD [ "bash" ]
