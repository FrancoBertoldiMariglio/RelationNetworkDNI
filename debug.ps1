# Set CUDA environment variables for debugging
$env:CUDA_LAUNCH_BLOCKING = "1"       # Fuerza ejecución síncrona para mejor debug
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"           # Evita buffering de la salida para ver prints inmediatamente
$env:TORCH_DISTRIBUTED_DEBUG = "DETAIL" # Más información de distributed training si se usa

# Create timestamp for debug log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logfile = "debug_log_${timestamp}.log"

# Run the training script with minimal data for debugging
python -m pdb main.py `
   --valid_dir data/train/valid `
   --invalid_dir data/train/invalid `
   --test_dir_valid data/test/valid `
   --test_dir_invalid data/test/invalid `
   --epochs 2 `
   --episodes_per_epoch 5 `
   --n_shot 5 `
   --n_query 15 `
   --learning_rate 0.001 `
   --batch_size 4 `
   --num_workers 0 `
   --hidden_size 8 `
   --dropout 0.1 `
   --seed 42 `
   --prefetch_factor 1 `
   --save_interval 1 | 
   Tee-Object -FilePath $logfile

# Add pause at the end to keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
   Write-Host "Press any key to continue..."
   $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}