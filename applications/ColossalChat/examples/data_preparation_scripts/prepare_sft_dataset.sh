SAVE_DIR="/gemini/code/ColossalAI/applications/ColossalChat/Mydataset"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python /gemini/code/ColossalAI/applications/ColossalChat/examples/data_preparation_scripts/prepare_dataset.py --type sft \
    --data_input_dirs /gemini/code/ColossalAI/applications/ColossalChat/SFT \
    --conversation_template_config /gemini/code/ColossalAI/applications/ColossalChat/config/conversation_template/chatGLM2.json \
    --tokenizer_dir  "" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
