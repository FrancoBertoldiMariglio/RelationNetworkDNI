# Set CUDA environment variables
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256"
$env:CUDA_VISIBLE_DEVICES = "0"

# Create timestamp for log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logfile = "training_log_${timestamp}.log"

# Run the training script
python main.py `
    --valid_dir data/train/valid `
    --invalid_dir data/train/invalid `
    --test_dir_valid data/test/valid `
    --test_dir_invalid data/test/invalid `
    --epochs 100 `
    --episodes_per_epoch 100 `
    --n_shot 5 `
    --n_query 15 `
    --learning_rate 0.001 `
    --batch_size 16 `
    --num_workers 4 `
    --hidden_size 8 `
    --dropout 0.1 `
    --seed 42 `
    --prefetch_factor 1 `
    --save_interval 5 | 
    Tee-Object -FilePath $logfile

# Add pause at the end to keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}