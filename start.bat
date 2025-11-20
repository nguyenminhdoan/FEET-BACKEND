@echo off
echo ======================================
echo Fleet Backend API
echo Starting Production Server
echo ======================================
echo.

docker-compose up -d --build

echo.
echo ======================================
echo Backend API started successfully!
echo ======================================
echo.
echo Access the API at:
echo   - API Endpoint: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo   - Health Check: http://localhost:8000/api/health
echo.
echo To view logs:
echo   docker-compose logs -f
echo.
echo To stop the service:
echo   docker-compose down
echo.
pause
