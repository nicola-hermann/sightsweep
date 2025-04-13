#!/bin/bash
# --- Configuration ---
# Number of processes to run in parallel.
NUM_PROC=6

# --- Command Line Arguments ---
# Dataset split to download. Options: train, test, index.
SPLIT=$1

# Inclusive upper limit for file downloads. Should be set according to split:
# train --> 499.
# test --> 19.
# index --> 99.
N=$2

# --- Input Validation ---
if [[ -z "$SPLIT" || -z "$N" ]]; then
  echo "Usage: $0 <split> <N>"
  echo "  split: train, test, or index"
  echo "  N: Upper limit (e.g., 499 for train)"
  exit 1
fi

# Check if N is a non-negative integer
if ! [[ "$N" =~ ^[0-9]+$ ]]; then
   echo "Error: N must be a non-negative integer."
   exit 1
fi
# --- End Input Validation ---

# --- Function Definition ---
# Downloads and extracts a single tar file.
download_and_extract() {
  # $1 is the formatted index number (e.g., 001, 042)
  local index_str=$1
  local images_file_name="images_${index_str}.tar"
  local images_tar_url="https://s3.amazonaws.com/google-landmark/$SPLIT/$images_file_name"

  echo "[${index_str}] Downloading $images_file_name..."
  # Use curl: -O saves with original filename, -s is silent (no progress meter)
  # Redirect standard output to /dev/null to hide download success message (errors still show on stderr)
  if curl -Os "$images_tar_url" > /dev/null; then
    echo "[${index_str}] Download complete. Extracting $images_file_name..."
    # Use tar: x=extract, f=file. ./ specifies the current directory explicitly.
    if tar -xf "./$images_file_name"; then
      echo "[${index_str}] Successfully extracted $images_file_name."
      # Optional: uncomment the next line to remove the tar file after successful extraction
      # rm "./$images_file_name"
      # echo "[${index_str}] Removed $images_file_name."
    else
      # Log tar extraction error to stderr
      echo "[${index_str}] Error: Failed to extract $images_file_name." >&2
    fi
  else
    # Log curl download error to stderr
    echo "[${index_str}] Error: Failed to download $images_file_name from $images_tar_url" >&2
  fi
}

# Export the function so it's available to the subshells created by '&'
# This is crucial for the parallel execution to work correctly in Bash.
export -f download_and_extract

# --- Main Execution Logic ---
echo "Starting download and extraction process..."
echo "Dataset Split: $SPLIT"
echo "Upper Limit (N): $N"
echo "Parallel Processes: $NUM_PROC"
echo "-------------------------------------"

# Loop from 0 up to N, processing in batches of size NUM_PROC
for i in $(seq 0 $NUM_PROC $N); do
  # Calculate the upper bound for this batch, ensuring it doesn't exceed N
  upper=$(expr $i + $NUM_PROC - 1)
  limit=$(($upper > $N ? $N : $upper)) # Use bash arithmetic conditional

  echo "--- Processing batch: Files $i to $limit ---"

  # Inner loop for the current batch
  # seq -f "%03g" formats the number with leading zeros (e.g., 000, 001, ..., 009, 010, ...)
  for j in $(seq -f "%03g" $i $limit); do
    # Call the function for each file index 'j' and run it in the background (&)
    download_and_extract "$j" &
  done

  # Wait for all background jobs *in the current batch* to complete
  # before starting the next batch.
  wait
  echo "--- Batch $i to $limit finished ---"

done

echo "-------------------------------------"
echo "All download and extraction tasks initiated. Check logs for errors."
# Note: The script finishes when the loops are done, but background processes
# from the *last* batch might still be running. The final 'wait' ensures those finish.
wait
echo "All processes have completed."