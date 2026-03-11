@echo off
REM Validation script to check if all required files exist
REM Run this before attempting to train

echo ============================================================
echo IMPLEMENTATION VALIDATION CHECK
echo ============================================================
echo.

set ERROR_COUNT=0

echo [1/11] Checking transformer_model.py...
if exist "transformer_model.py" (
    echo     [OK] Found transformer_model.py
) else (
    echo     [ERROR] Missing transformer_model.py
    set /a ERROR_COUNT+=1
)

echo [2/11] Checking triplet_loss.py...
if exist "triplet_loss.py" (
    echo     [OK] Found triplet_loss.py
) else (
    echo     [ERROR] Missing triplet_loss.py
    set /a ERROR_COUNT+=1
)

echo [3/11] Checking data_utils.py...
if exist "data_utils.py" (
    echo     [OK] Found data_utils.py
) else (
    echo     [ERROR] Missing data_utils.py
    set /a ERROR_COUNT+=1
)

echo [4/11] Checking train.py...
if exist "train.py" (
    echo     [OK] Found train.py
) else (
    echo     [ERROR] Missing train.py
    set /a ERROR_COUNT+=1
)

echo [5/11] Checking inference.py...
if exist "inference.py" (
    echo     [OK] Found inference.py
) else (
    echo     [ERROR] Missing inference.py
    set /a ERROR_COUNT+=1
)

echo [6/11] Checking requirements.txt...
if exist "requirements.txt" (
    echo     [OK] Found requirements.txt
) else (
    echo     [ERROR] Missing requirements.txt
    set /a ERROR_COUNT+=1
)

echo [7/11] Checking test_all.py...
if exist "test_all.py" (
    echo     [OK] Found test_all.py
) else (
    echo     [ERROR] Missing test_all.py
    set /a ERROR_COUNT+=1
)

echo [8/11] Checking quick_train.py...
if exist "quick_train.py" (
    echo     [OK] Found quick_train.py
) else (
    echo     [ERROR] Missing quick_train.py
    set /a ERROR_COUNT+=1
)

echo [9/11] Checking README.md...
if exist "README.md" (
    echo     [OK] Found README.md
) else (
    echo     [ERROR] Missing README.md
    set /a ERROR_COUNT+=1
)

echo [10/11] Checking GETTING_STARTED.py...
if exist "GETTING_STARTED.py" (
    echo     [OK] Found GETTING_STARTED.py
) else (
    echo     [ERROR] Missing GETTING_STARTED.py
    set /a ERROR_COUNT+=1
)

echo [11/11] Checking __init__.py...
if exist "__init__.py" (
    echo     [OK] Found __init__.py
) else (
    echo     [ERROR] Missing __init__.py
    set /a ERROR_COUNT+=1
)

echo.
echo ============================================================
echo CHECKING DATA DIRECTORY
echo ============================================================
echo.

if exist "..\data\" (
    echo [OK] Data directory exists: ..\data\
    
    if exist "..\data\train\" (
        echo [OK] Training data found
    ) else (
        echo [WARNING] Training data not found in ..\data\train\
    )
    
    if exist "..\data\validation\" (
        echo [OK] Validation data found
    ) else (
        echo [WARNING] Validation data not found in ..\data\validation\
    )
    
    if exist "..\data\test\" (
        echo [OK] Test data found
    ) else (
        echo [WARNING] Test data not found in ..\data\test\
    )
) else (
    echo [WARNING] Data directory not found: ..\data\
    echo           Download data first before training
)

echo.
echo ============================================================
echo VALIDATION SUMMARY
echo ============================================================
echo.

if %ERROR_COUNT%==0 (
    echo [SUCCESS] All required files are present!
    echo.
    echo You are ready to train. Next steps:
    echo   1. Install dependencies: pip install -r requirements.txt
    echo   2. Run tests: python test_all.py
    echo   3. Quick training: python quick_train.py
    echo   4. Full training: python train.py --data_dir ../data
    echo.
    exit /b 0
) else (
    echo [ERROR] %ERROR_COUNT% file(s) missing!
    echo Please ensure all files are created properly.
    echo.
    exit /b 1
)
