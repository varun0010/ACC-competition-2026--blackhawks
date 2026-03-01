@echo off
title Team Black Hawks - Full Ignition System
color 0A

echo =============================================================
echo   TEAM BLACK HAWKS -- FULL IGNITION SEQUENCE
echo =============================================================
echo.
echo ACTION REQUIRED BEFORE CONTINUING:
echo   1. Open Quanser Interactive Labs (QLabs)
echo   2. Select "Self-Driving Car Studio"
echo   3. Click "Studio" and wait for the 3D map to FULLY load
echo   4. Confirm the QCar2 vehicle is visible on the track
echo.
echo Press any key ONLY when the map is fully loaded...
pause > nul

:: ── Step 1: Activate virtual environment ─────────────────────────────────
echo.
echo [1/4] Activating Python environment...
call qcar_agent\Scripts\activate
if errorlevel 1 (
    echo ERROR: Virtual environment not found at qcar_agent\Scripts\activate
    echo Make sure you are running this bat from the ACC_Competition folder.
    pause
    exit /b 1
)
echo       OK - Environment active.

:: ── Step 2: Spawn world actors + traffic lights ───────────────────────────
echo.
echo [2/4] Spawning world actors and traffic lights...
start /b python scenario.py

:: Wait for QLabs actors to fully spawn before starting RT model.
:: scenario.py connects to QLabs, destroys all actors, then respawns them.
:: 10 seconds is the safe minimum for this on your hardware.
echo       Waiting 10 seconds for world to stabilise...
timeout /t 10 /nobreak > nul
echo       OK - World ready.

:: ── Step 3: Launch QUARC real-time model ─────────────────────────────────
echo.
echo [3/4] Starting QUARC Real-Time server (HIL model)...
start "QUARC Server" "C:\Program Files\Quanser\Quanser SDK\quarc_run.exe" -D -r -t tcpip://localhost:17000 "C:\Users\91778\Downloads\Q-car\Q-car\Quanser_Academic_Resources-dev-windows\Quanser_Academic_Resources-dev-windows\0_libraries\resources\rt_models\QCar2\QCar2_Workspace_studio.rt-win64"

:: CRITICAL WAIT: The HIL card (t_card handle) is not valid until the RT model
:: has fully initialised its hardware interface. The "invalid card" error
:: happens because the old 3-second wait is not long enough.
:: 15 seconds is the safe minimum for the RT model to reach RUNNING state.
echo       Waiting 15 seconds for QUARC RT model to reach RUNNING state...
echo       (Do NOT close the QUARC Server window that just opened)
timeout /t 15 /nobreak > nul

:: Extra confirmation prompt - verify QUARC window shows "Running"
echo.
echo ---------------------------------------------------------------
echo VERIFY: The QUARC Server window should now show "Running".
echo If it shows "Starting" or any error, wait longer or restart.
echo Press any key when QUARC status confirms RUNNING...
echo ---------------------------------------------------------------
pause > nul

:: ── Step 4: Launch the autonomous master agent ────────────────────────────
echo.
echo [4/4] Launching Black Hawks Master Agent...
echo       Controls: Q in OpenCV window to quit cleanly.
echo.
python acc_master_agent.py

:: ── Shutdown ──────────────────────────────────────────────────────────────
echo.
echo =============================================================
echo   Master agent exited. QUARC server is still running.
echo   Close the QUARC Server window manually if needed.
echo =============================================================
pause