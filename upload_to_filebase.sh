#!/bin/bash
# upload_to_filebase.sh
# Syncs local dataset to Filebase, only uploading new/changed files

LOCAL_PATH="local_storage/datasets/turbulent_radiative_layer_2D/"
S3_PATH="s3://graph-wavelet-compressed-sensing/datasets/turbulent_radiative_layer_2D/"
ENDPOINT="https://s3.filebase.com"

echo "Syncing files from:"
echo "  Local:  $LOCAL_PATH"
echo "  Remote: $S3_PATH"
echo ""

aws --endpoint-url "$ENDPOINT" s3 sync "$LOCAL_PATH" "$S3_PATH"

echo ""
echo "Sync complete!"


LOCAL_PATH="local_storage/runs_turbulent/"
S3_PATH="s3://graph-wavelet-compressed-sensing/runs_turbulent/"
ENDPOINT="https://s3.filebase.com"

echo "Syncing files from:"
echo "  Local:  $LOCAL_PATH"
echo "  Remote: $S3_PATH"
echo ""

aws --endpoint-url "$ENDPOINT" s3 sync "$LOCAL_PATH" "$S3_PATH"

echo ""
echo "Sync complete!"