Write-Host "Running..."

# Activate the virtual environment
.\venv\Scripts\activate
$lamValues = 0.5, 0.6, 0.7, 0.8

foreach ($lam in $lamValues) {
    Write-Host "Running for lam = $lam"

    $process = Start-Process python -ArgumentList "pepsflow/start_pepsflow.py", `
        "--chi", "6", `
        "--D", "2", `
        "--lam", "$lam", `
        "--max_iter", "20", `
        "--runs", "1", `
        "--lr", "1", `
        "--epochs", "10", `
        "--perturbation", "0.0", `
        "--fn", "tests/test_lam_$lam" `
        -NoNewWindow -RedirectStandardOutput output.txt -PassThru

    $process | Wait-Process
}

Write-Host "All tasks completed."


