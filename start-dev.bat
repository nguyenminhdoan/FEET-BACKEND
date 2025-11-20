@echo off
echo ======================================
echo Fleet Backend API
echo Starting Development Server
echo ======================================
echo.

docker-compose -f docker-compose.dev.yml up --build

echo.
echo ======================================
echo Backend API stopped
echo ======================================
pause
