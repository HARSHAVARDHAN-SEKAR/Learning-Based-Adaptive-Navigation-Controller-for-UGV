# Reproducible environment for the Python research layer (Stages 1-6).
# NOT the ROS2 deployment image — see ros2_ws/Dockerfile for that.
#
# Build:  docker build -t ugv-nav-research .
# Run:    docker run --rm -v "$(pwd)/benchmarks:/app/benchmarks" ugv-nav-research
#         (runs the full benchmark suite and writes results back to the host)

FROM python:3.11-slim

# ipopt (via casadi wheel) needs libgomp; matplotlib headless needs no X11 libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the build if the code itself is broken — makes this Dockerfile
# double as a CI gate, not just a runtime image.
RUN python3 tests/test_all_modules.py

ENV MPLBACKEND=Agg
ENTRYPOINT ["python3"]
CMD ["benchmarks/run_full_flow.py"]
