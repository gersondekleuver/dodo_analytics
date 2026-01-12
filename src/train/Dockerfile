FROM ultralytics/ultralytics:latest

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# print message
RUN echo "Installing dependencies"

# Copy the application code
COPY train.py ./
COPY labels.yaml ./

# Setup an app user so the container doesn't run as the root user
CMD ["python", "train.py", "-m", "yolo11m.pt"]


# set 1
# docker run -it --rm --gpus all --mount type=bind,source=C:/Users/kleuv/Productie/dodo_ai/dodo_analytics/runs,target=/ultralytics/runs --mount type=bind,source=C:/Users/kleuv/Productie/dodo_ai/dodo_analytics/datasets/bbox/unified_dataset,target=/ultralytics/datasets/bbox/unified_dataset dodo


# /ultralytics/datasets/bbox/unified_dataset
# docker run -it --rm --gpus all --mount type=bind,source=/home/gersondekleuver/runs,target=/ultralytics/runs --mount type=bind,source=/home/gersondekleuver/unified_dataset_small,target=/ultralytics/datasets/bbox/unified_dataset dodo

# docker build . --rm --tag "dodo"
